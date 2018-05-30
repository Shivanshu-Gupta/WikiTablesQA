package org.allenai.pnp

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer
import com.google.common.base.Preconditions
import com.jayantkrish.jklol.training.LogFunction
import edu.cmu.dynet._
import org.allenai.pnp.semparse.SemanticParserState
import org.allenai.wikitables.WikiTablesExample

import scala.util.Random

class LoglikelihoodTrainer(val epochs: Int, val beamSize: Int, val sumMultipleExecutions: Boolean,
    val model: PnpModel, val trainer: Trainer, val log: LogFunction) {

  Preconditions.checkArgument(model.locallyNormalized == true)

  def train[A](examples: Seq[PnpExample[A]], wikiexamples: Seq[WikiTablesExample] = null): Unit = {
    for (i <- 0 until epochs) {
      var loss = 0.0
      var searchErrors = 0
      log.notifyIterationStart(i)

      log.startTimer("pp_loglikelihood")
      for ((example, wikiexample) <- Random.shuffle(examples zip wikiexamples)) {
        ComputationGraph.renew()
        // println(wikiexample.id)
        val env = example.env
        val context = PnpInferenceContext.init(model).setLog(log)

        // Compute the distribution over correct executions.
        log.startTimer("pp_loglikelihood/forward")
        val conditional = example.conditional.beamSearch(beamSize, -1,
          env, context.addExecutionScore(example.conditionalExecutionScore))
        log.stopTimer("pp_loglikelihood/forward")
        
        log.startTimer("pp_loglikelihood/build_loss")
        val exLosses = conditional.executions.map(_.env.getScore)

        val states = conditional.executions.map(_.value.asInstanceOf[SemanticParserState])
        val covs = states.map(_.getAttCoverage())
        val scores = states.map(_.getScoreMatrix())
        // KL Divergence
//        val probScore = Expression.transpose(Expression.softmax(Expression.transpose(score)))
        val probScores = for{
          score <- scores
        } yield {
          Expression.transpose(Expression.softmax(Expression.transpose(score)))
        }

//        val logProbScore = Expression.log(probScore)
        val logProbScores = for{
          probScore <- probScores
        } yield {
          Expression.log(probScore)
        }

//        val prodScore = Expression.cmult(probScore, logProbScore)
        val prodScores = for{
          (probScore, logProbScore) <- probScores zip logProbScores
        } yield {
          Expression.cmult(probScore, logProbScore)
        }

//        val imp = Expression.sumCols(prodScore)
        val impScores = for{
          prodScore <- prodScores
        } yield {
          Expression.sumCols(prodScore)
        }

        println("Dummy")

        val coverage = for{
          (impScore, cov) <- impScores zip covs
        }  yield {
          Expression.cmult(Expression.rectify(impScore - Expression.averageCols(Expression.transpose(impScore))), cov)
        }

        val logProbExpr = if (exLosses.isEmpty) {
          Preconditions.checkState(sumMultipleExecutions,
            "Found %s conditional executions (expected exactly 1) for example: %s",
            conditional.executions.size.asInstanceOf[AnyRef], example)

          null
        } else if (exLosses.length == 1) {
          exLosses(0)
        } else {
          // This flag is used to ensure that training with a
          // single label per example doesn't work "by accident"
          // with an execution score that permits multiple labels.
          Preconditions.checkState(sumMultipleExecutions,
            "Found %s conditional executions (expected exactly 1) for example: %s",
            conditional.executions.size.asInstanceOf[AnyRef], example)

//          Expression.logSumExp(new ExpressionVector(exLosses))
          Expression.sum(Expression.logSumExp(new ExpressionVector(exLosses)),0.01f * Expression.sumRows(Expression.sum(impScores:_*)))
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
        }
        else {
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
