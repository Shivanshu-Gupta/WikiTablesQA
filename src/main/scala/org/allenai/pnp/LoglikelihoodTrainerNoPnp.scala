package org.allenai.pnp

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer
import com.google.common.base.Preconditions
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.training.LogFunction
import edu.cmu.dynet._
import org.allenai.pnp.semparse.{EntityLinking, SemanticParserNoPnp}
import org.allenai.wikitables.WikiTablesExample

import scala.util.Random

class LoglikelihoodTrainerNoPnp(val epochs: Int, val sumMultipleExecutions: Boolean, val parser: SemanticParserNoPnp,
                                val trainer: Trainer, val log: LogFunction, val typeDeclaration: TypeDeclaration,
                                val k: Int = -1, val margin: Int = -1) {

  def getLossExpr(exid: String, exLosses: Seq[Expression]) = {
    val sortedLosses = exLosses.sortBy(ComputationGraph.incrementalForward(_).toFloat).reverse
    if (k == 1) {
      if (margin == -1) {
        if (!exid.contains('_') || exid.split('_')(1).startsWith("p")) {
          sortedLosses.head
        } else {
          Expression.log(1 - Expression.exp(sortedLosses.last))
        }
      } else {
        Expression.log(Expression.exp(sortedLosses.head) + 1 - Expression.exp(sortedLosses.last))
      }
    } else {
      if (margin == -1) {
        if (!exid.contains('_') || exid.split('_')(1).startsWith("p")) {
          val corr = sortedLosses.take(k)
          Expression.logSumExp(new ExpressionVector(corr))
        } else {
          val incorr = sortedLosses.reverse.take(k)
          Expression.log(k - Expression.sum(new ExpressionVector(incorr.map(Expression.exp))))
        }
      } else {
        if (sortedLosses.length >= k + margin + k) {
          val corr = sortedLosses.take(k)
          val incorr = sortedLosses.slice(k + margin, k + margin + k)
          Expression.log(Expression.sum(new ExpressionVector(corr.map(Expression.exp)))
            + k - Expression.sum(new ExpressionVector(incorr.map(Expression.exp))))
        } else {
          val corr = sortedLosses.take(Math.min(k, sortedLosses.length / 2))
          val incorr = sortedLosses.reverse.take(Math.min(k, sortedLosses.length / 2))
          Expression.log(Expression.sum(new ExpressionVector(corr.map(Expression.exp)))
            + incorr.length - Expression.sum(new ExpressionVector(incorr.map(Expression.exp))))
        }
      }
    }
  }

  def train[A](wikiexamples: Seq[WikiTablesExample] = null): Unit = {
    wikiexamples.foreach { ex =>
      val entityLinking = ex.sentence.getAnnotation("entityLinking").asInstanceOf[EntityLinking]
      ex.groupedTemplateSeqs = ex.logicalForms.toList.flatMap {lf =>
        parser.generateActionSequence(lf, entityLinking, typeDeclaration).map(y => (lf, y._2))
      }.groupBy{lfTemplatePair =>
        val typeMap = StaticAnalysis.inferTypeMap(lfTemplatePair._1, TypeDeclaration.TOP, typeDeclaration)
          .asScala.toMap
        typeMap(0)
      }.toList
    }
    val examples = wikiexamples.filter(_.groupedTemplateSeqs.nonEmpty)
    println(examples.size +  " training examples after action sequence generation.")
    for (i <- 0 until epochs) {
      var loss = 0.0
      var searchErrors = 0
      log.notifyIterationStart(i)

      log.startTimer("pp_loglikelihood")
      for(example <- Random.shuffle(examples)) {
        try {
          ComputationGraph.renew( )
          // Compute the distribution over correct executions.
          log.startTimer("pp_loglikelihood/forward")
          val results = parser.generateLogProbs(example)
          log.stopTimer("pp_loglikelihood/forward")

          log.startTimer("pp_loglikelihood/build_exloss")
          val exLosses = results.map(_._2)
          val logProbExpr = if (exLosses.isEmpty) {
            Preconditions.checkState(sumMultipleExecutions,
              "Found %s logical forms (expected exactly 1) for example: %s",
              exLosses.length.asInstanceOf[AnyRef], example)
            null
          } else if (exLosses.length == 1) {
            exLosses.head
          } else {
            // This flag is used to ensure that training with a
            // single label per example doesn't work "by accident"
            // with an execution score that permits multiple labels.
            Preconditions.checkState(sumMultipleExecutions,
              "Found %s logical forms (expected exactly 1) for example: %s",
              exLosses.size.asInstanceOf[AnyRef], example)
            if (k == -1) {
              Expression.logSumExp(new ExpressionVector(exLosses))
            } else {
              getLossExpr(example.id, exLosses)
            }
          }
          log.stopTimer("pp_loglikelihood/build_exloss")
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
        } catch {
          case e: Throwable =>
            println(s"Exception with example: ${example.id} with ")
            e.printStackTrace()

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
