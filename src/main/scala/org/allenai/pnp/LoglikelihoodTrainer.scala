package org.allenai.pnp

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer
import com.google.common.base.Preconditions
import com.jayantkrish.jklol.training.LogFunction
import edu.cmu.dynet._
import org.allenai.wikitables.WikiTablesExample

import scala.util.Random

class LoglikelihoodTrainer(val epochs: Int, val beamSize: Int, val sumMultipleExecutions: Boolean,
    val model: PnpModel, val trainer: Trainer, val log: LogFunction, val k: Int = -1, val margin: Int = -1) {

  Preconditions.checkArgument(model.locallyNormalized == true)

  def train[A](examples: Seq[PnpExample[A]], wikiexamples: Seq[WikiTablesExample] = null): Unit = {
    for (i <- 0 until epochs) {
      var loss = 0.0
      var searchErrors = 0
      log.notifyIterationStart(i)

      log.startTimer("pp_loglikelihood")
      for ((wikiexample, example)<- Random.shuffle(wikiexamples zip examples)) {
        ComputationGraph.renew()

        val env = example.env
        val context = PnpInferenceContext.init(model).setLog(log)

        // Compute the distribution over correct executions.
        log.startTimer("pp_loglikelihood/forward")
        val conditional = example.conditional.beamSearch(beamSize, -1,
          env, context.addExecutionScore(example.conditionalExecutionScore))
        log.stopTimer("pp_loglikelihood/forward")
        
        log.startTimer("pp_loglikelihood/build_loss")
        val exLosses = conditional.executions.map(_.env.getScore)

        val logProbExpr = if (exLosses.isEmpty) {
          Preconditions.checkState(sumMultipleExecutions,
            "Found %s conditional executions (expected exactly 1) for example: %s",
            conditional.executions.size.asInstanceOf[AnyRef], example)

          null
        } else if (exLosses.length == 1) {
          exLosses.head
        } else {
          // This flag is used to ensure that training with a
          // single label per example doesn't work "by accident" 
          // with an execution score that permits multiple labels.
          Preconditions.checkState(sumMultipleExecutions,
            "Found %s conditional executions (expected exactly 1) for example: %s",
            conditional.executions.size.asInstanceOf[AnyRef], example)

          if(k == -1) {
            Expression.logSumExp(new ExpressionVector(exLosses))
          } else {
            val sortedLosses = exLosses.sortBy(ComputationGraph.incrementalForward(_).toFloat)
            if (k == 1) {
              if (margin == -1) {
                if(wikiexample.id.split('_')(1).startsWith("p")) {
                  sortedLosses.last
                } else {
                  -sortedLosses.head
                }
              } else {
                sortedLosses.last - sortedLosses.head
              }
            } else {
              if(margin == -1) {
                if(wikiexample.id.split('_')(1).startsWith("p")) {
                  val corr = sortedLosses.take(k)
                  Expression.logSumExp(new ExpressionVector(corr))
                } else {
                  val incorr = sortedLosses.reverse.take(k)
                  Expression.log(-Expression.sum(new ExpressionVector(incorr.map(Expression.exp))))
                }
              } else {
                val corr = sortedLosses.take(k)
                val incorr = sortedLosses.reverse.slice(k + margin, k + margin + k)
                Expression.log(Expression.sum(new ExpressionVector(corr.map(Expression.exp)))
                  - Expression.sum(new ExpressionVector(incorr.map(Expression.exp))))
              }
            }
          }
        }
        log.stopTimer("pp_loglikelihood/build_loss")

        val lossExpr = -1.0f * logProbExpr
        if (lossExpr != null) {
          log.startTimer("pp_loglikelihood/eval_loss")
          loss += ComputationGraph.incrementalForward(lossExpr).toFloat
          log.stopTimer("pp_loglikelihood/eval_loss")

          // cg.print_graphviz()
          log.startTimer("pp_loglikelihood/backward")
          ComputationGraph.backward(lossExpr)
          trainer.update(1.0f)
          log.stopTimer("pp_loglikelihood/backward")
        } else {
          searchErrors += 1
        }
      }
      log.stopTimer("pp_loglikelihood")
      
      trainer.updateEpoch()

      log.logStatistic(i, "loss", loss)
      log.logStatistic(i, "search errors", searchErrors)
      log.notifyIterationEnd(i)
    }
  }
}
