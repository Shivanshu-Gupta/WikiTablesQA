package org.allenai.pnp

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer
import com.google.common.base.Preconditions
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.training.LogFunction
import edu.cmu.dynet._
import org.allenai.pnp.semparse.{EntityLinking, SemanticParserState}
import org.allenai.wikitables.WikiTablesExample

import scala.util.Random
import scala.collection.JavaConversions._

class LoglikelihoodTrainer(val epochs: Int, val beamSize: Int, val sumMultipleExecutions: Boolean,
    val model: PnpModel, val trainer: Trainer, val log: LogFunction) {

  Preconditions.checkArgument(model.locallyNormalized == true)

  def getEntities(expr : Expression2) : ListBuffer[String] = {
    val subExprList = expr.getSubexpressions()
    var entitiesList = new ListBuffer[String]()
    for(subExpr <- subExprList) {
      if(subExpr.isConstant){
        entitiesList += subExpr.getConstant
      } else {
        entitiesList ++= getEntities(subExpr)
      }
    }
    entitiesList
  }

  def getEntityIndices(entitiesBeam: List[ListBuffer[String]], entityLinking: EntityLinking): List[List[Int]] ={
    var beamIndicesList = ListBuffer[List[Int]]()
    for(entities <- entitiesBeam) {
      var indicesList = ListBuffer[Int]()
      for(entity <- entities) {
        val index = getEntityIndex(entity, entityLinking)
        if(index != -1) {
          indicesList += index
        }
      }
      beamIndicesList += indicesList.toList
    }
    beamIndicesList.toList
  }

  def getEntityIndex(entityString: String, entityLinking: EntityLinking): Int = {
    var foundIndex = -1

    for ((entity, i) <- entityLinking.entities.zipWithIndex) {
      if(entity.expr.toString() == entityString) {
        foundIndex = i
      }
    }
    foundIndex
  }

  def increaseScore(entitiesB: List[ListBuffer[String]], tokenEntityScoresB: List[Expression], entityLinking: EntityLinking): Expression = {
    /**
      * Increase the probability of these entities
      * Check which token has best score with this entity and increase the (token, entity) probability
      */
    var incProbB : Expression = null

    for((entities, i) <- entitiesB.zipWithIndex) {
      for (entity <- entities) {

        val entityIndex = getEntityIndex(entity, entityLinking)
        if(entityIndex != -1){
          val entityScores = Expression.pick(tokenEntityScoresB(i), entityIndex, 1)
          val maxTokenIndex = ComputationGraph.forward(entityScores).toVector().zipWithIndex.maxBy(_._1)._2
          val incProb = Expression.pick(entityScores, maxTokenIndex)
          if(incProbB == null){
            incProbB = incProb
          } else {
            incProbB = incProbB + incProb
          }
        }

      }
    }
    incProbB
  }

  def decreaseScore(entitiesB: List[ListBuffer[String]], tokenEntityScoresB: List[Expression], entityLinking: EntityLinking, tokens: List[String]): Expression = {
    /**
      * Decrease the scores of those tokens whose max is not in the list of entities
      * Find the max entity of the token and if it is not in list of entities decrease the score of (Token, Entity) pair
      */
    var decProbB : Expression = null
    var nullIndex = getEntityIndex("-1000", entityLinking)

    val entityIndicesB = getEntityIndices(entitiesB, entityLinking)
    for((entityIndices, i) <- entityIndicesB.zipWithIndex) {
      for((token, j) <- tokens.zipWithIndex){
        val tokenScores = Expression.pick(tokenEntityScoresB(i), j, 0) // Pick the scores of the jth token
        val maxEntityIndex = ComputationGraph.forward(tokenScores).toVector().zipWithIndex.maxBy(_._1)._2
        var decProb : Expression = null
        if(! entityIndices.contains(maxEntityIndex) && maxEntityIndex != nullIndex) {
           decProb = Expression.pick(tokenScores, maxEntityIndex)
        }
        if(decProb != null) {
          if(decProbB == null) {
            decProbB = decProb
          } else {
            decProbB = decProbB + decProb
          }
        }
      }
    }
    decProbB
  }

  def getSumEntityExpr(tokenEntityScoresBeam: List[Expression], entityIndicesBeam: List[List[Int]]): List[Expression] = {
    var sumExpList = new ListBuffer[Expression]()
    for((tes, entityIndices) <- tokenEntityScoresBeam.zip(entityIndicesBeam)) {
      val teps = Expression.transpose(Expression.softmax(Expression.transpose(tes))) // token entity probability scores
      var sumExp : Expression = null
      for(index <- entityIndices) {
        val exp = Expression.sumRows(Expression.pick(teps, index, 1)) // 1 stands for the dimension
        if(sumExp == null) {
          sumExp = exp
        } else {
          sumExp = Expression.sum(sumExp, exp)
        }
      }
      sumExpList += sumExp
    }
    sumExpList.toList
  }

  def train[A](examples: Seq[PnpExample[A]], wikiExamples: Seq[WikiTablesExample] = Nil, entityLinkings: Seq[EntityLinking] = Nil,
               elRegulizer: Double = 0.0): Unit = {
    for (i <- 0 until epochs) {
      var loss = 0.0
      var searchErrors = 0
      log.notifyIterationStart(i)

      log.startTimer("pp_loglikelihood")
      for ((example, wikiExample, entityLinking) <- Random.shuffle((examples, wikiExamples, entityLinkings).zipped.toList)) {
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

        val states = conditional.executions.map(_.value.asInstanceOf[SemanticParserState])
        val tokenEntityScores = states.map(_.getScoreMatrix())

        val expressions = states.map(_.decodeExpression)
        val entities = expressions.map(getEntities(_))
        //        val indicesListBeam = getEntityIndices(entities.toList, entityLinking)
        //        val sumEntityExpr = getSumEntityExpr(tokenEntityScores.toList, indicesListBeam)
        val incExpr = increaseScore(entities.toList, tokenEntityScores.toList, entityLinking)
//        val decExpr = decreaseScore(entities.toList, tokenEntityScores.toList, entityLinking, wikiExample.sentence.getWords.toList)

        val logProbExpr = if (exLosses.length == 0) {
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

          Expression.logSumExp(new ExpressionVector(exLosses))
        }

        //        val entityExprFiltered = entityExpr.filter(e => e != null)
        //        if(entityExprFiltered.size != 0 && i > 3){
        //         logProbEntityExpr = logProbExpr + elRegulizer * Expression.logSumExp(new ExpressionVector(entityExprFiltered))
        //      }
        var logProbAugExpr = logProbExpr
        if(i > 1){
          if(incExpr != null){
            logProbAugExpr = logProbAugExpr +  elRegulizer * incExpr
          }
//          if(decExpr != null){
//            logProbAugExpr = logProbAugExpr - elRegulizer * decExpr
//          }
        }

//        if(ComputationGraph.forward(logProbAugExpr).toVector().get(0).isNaN()) {
//          val issue = true
//        }
        val lossExpr = -1.0f * logProbAugExpr

        log.stopTimer("pp_loglikelihood/build_loss")

//        val lossExpr = -1.0f * logProbExpr
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
