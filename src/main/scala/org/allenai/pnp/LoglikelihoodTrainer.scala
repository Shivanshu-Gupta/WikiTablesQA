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
  def eye(t: Int): Expression = {
      val matrix = new FloatVector(t * t)
      for(i <- 0 until t) {
        matrix(t * i + i) = 1
      }
    Expression.input(Dim(t, t), matrix)
  }

  def getAttentionLosses(states: Seq[SemanticParserState], attentionLoss: String): Seq[Expression] = {
    for {
      state <- states
      templateTypes = state.getTemplateTypes
      attentions = if(attentionLoss == "entityTemplates") {
        state.getAttentions.zipWithIndex.filter(x => templateTypes(x._2) == "entity").map(_._1)
      } else {
        state.getAttentions
      }
      normalized = attentions.map(x => Expression.exprTimes(Expression.pick(x, 0), Expression.inverse(Expression.sqrt(Expression.squaredNorm(x)))))
      if normalized.length > 0
      attentionMatrix = Expression.concatenateCols(new ExpressionVector(normalized))
      identity = eye(normalized.length)
    } yield {
      Expression.squaredNorm(Expression.transpose(attentionMatrix) * attentionMatrix - identity)
    }
  }

  def train[A](examples: Seq[PnpExample[A]], wikiexamples: Seq[WikiTablesExample] = null, attentionLossParams: String = null): Unit = {
    for (i <- 0 until epochs) {
      var loss = 0.0
      var searchErrors = 0
      log.notifyIterationStart(i)

      log.startTimer("pp_loglikelihood")
      for ((example, wikiexample) <- Random.shuffle(examples zip wikiexamples)) {
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
          Expression.logSumExp(new ExpressionVector(exLosses))
        }

        val attentionLossAddedExpr = if(logProbExpr != null && attentionLossParams != "") {
          val params = attentionLossParams.split(":")
          val attentionLossType = params.head
          val attentionLossWeight = params(1).toFloat
          val states = conditional.executions.map(_.value.asInstanceOf[SemanticParserState])
          val attentionLosses = getAttentionLosses(states, attentionLossType)
          if(attentionLosses.nonEmpty) {
            val attentionLoss = if(exLosses.length == 1) attentionLosses.head
                                else Expression.sum(new ExpressionVector(attentionLosses))
            Expression.sum(logProbExpr, -attentionLossWeight * attentionLoss)
          } else {
            logProbExpr
          }
        } else {
          logProbExpr
        }
        log.stopTimer("pp_loglikelihood/build_loss")

        val lossExpr = -1.0f * attentionLossAddedExpr
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
