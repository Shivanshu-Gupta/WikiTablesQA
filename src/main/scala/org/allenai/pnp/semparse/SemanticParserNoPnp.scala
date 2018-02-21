package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.{Map => MutableMap}
import scala.collection.mutable.MultiMap
import scala.collection.mutable.{Set => MutableSet}
import org.allenai.pnp.{CompGraph, Env, Pnp}
import org.allenai.pnp.ExecutionScore.ExecutionScore
import org.allenai.wikitables.WikiTablesExample
//import org.allenai.pnp.Pnp
//import org.allenai.pnp.PnpModel
import org.allenai.pnp.util.Trie
import org.allenai.wikitables.SemanticParserFeatureGenerator
import com.google.common.base.Preconditions
import com.jayantkrish.jklol.ccg.lambda.Type
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import com.jayantkrish.jklol.ccg.lambda2.StaticAnalysis
import com.jayantkrish.jklol.util.IndexedList
import edu.cmu.dynet._
import org.allenai.wikitables.LfPreprocessor

import scala.collection.mutable

/** A parser mapping token sequences to a distribution over
  * logical forms.
  */
class SemanticParserNoPnp(val actionSpace: ActionSpace, val vocab: IndexedList[String],
                          val config: SemanticParserConfig, forwardBuilder: LstmBuilder,
                          backwardBuilder: LstmBuilder, actionBuilder: LstmBuilder,
                          val model: Model, params: Map[String, Parameter],
                          lookupParams: Map[String, LookupParameter]) {

  var dropoutProb = -1.0

  import SemanticParserNoPnp._

  private def initializeRnns(): Unit = {
    forwardBuilder.newGraph()
    backwardBuilder.newGraph()
    actionBuilder.newGraph()
  }

  /** Compute the input encoding of a list of tokens
    */
  def encode(tokens: Array[Int], entityLinking: EntityLinking): InputEncoding = {
    initializeRnns()
    encodeInput(entityLinking, tokens)
  }

  private def encodeInput(entityLinking: EntityLinking, tokens: Array[Int]): InputEncoding = {

    val wordEmbeddings = lookupParams(SemanticParserNoPnp.WORD_EMBEDDINGS_PARAM)
    def tokenIdToEmbedding(tokenId: Int): Expression = {
      if (tokenId < vocab.size) {
        Expression.lookup(wordEmbeddings, tokenId)
      } else {
        Expression.lookup(wordEmbeddings, vocab.size)
      }
    }

    val entityEncoding = encodeEntities(entityLinking, tokens, tokenIdToEmbedding)
    val inputEncoding = rnnEncode(tokens, tokenIdToEmbedding, entityEncoding, entityLinking)
    InputEncoding(tokens, inputEncoding._1, inputEncoding._2, inputEncoding._3,
        inputEncoding._4, entityEncoding)
  }

  private def getTokenEmbeddings(tokens: Seq[Int], tokenIdToEmbedding: Int => Expression,
      entityEncoding: EntityEncoding): Array[Expression] = {
    val lookups = tokens.map(tokenIdToEmbedding(_)).toArray

    val entityVectors = tokens.map(x => Expression.input(Dim(config.entityDim),
        new FloatVector(List.fill(config.entityDim)(0.0f)))).toArray

    for ((t, m) <- entityEncoding.tokenEntityScoreMatrices) {
      val numRows = m.dim.rows().toInt
      val zeros = Expression.input(Dim(numRows), new FloatVector(List.fill(numRows)(0.0f)))
      val scores = Expression.concatenateCols(new ExpressionVector(List(m, zeros)))

      // A (entities + 1) x tokens matrix of softmax scores
      val softmaxedScores = Expression.softmax(Expression.transpose(scores))

      // columns of this matrix are entities
      val entityEmbeddings = entityEncoding.entityEmbeddingMatrices(t)
      val numEntityRows = entityEmbeddings.dim.rows().toInt
      val entityZeros = Expression.input(Dim(numEntityRows),
          new FloatVector(List.fill(numEntityRows)(0.0f)))
      val entityEmbeddingsWithZero = Expression.concatenateCols(
          new ExpressionVector(List(entityEmbeddings, entityZeros)))

      // An inputDim x tokens matrix
      val tokenEntityEmbeddings = entityEmbeddingsWithZero * softmaxedScores

      for (i <- entityVectors.indices) {
        entityVectors(i) += Expression.pick(tokenEntityEmbeddings, i, 1)
      }
    }

    lookups.zip(entityVectors).map(x => Expression.concatenate(new ExpressionVector(List(x._1, x._2))))
  }

  private def rnnEncode(tokens: Seq[Int], tokenIdToEmbedding: Int => Expression,
                        entityEncoding: EntityEncoding, entityLinking: EntityLinking):
  (ExpressionVector, Expression, Expression, Expression) = {
    import Expression.{ dropout, reshape }

    var inputEmbeddings = getTokenEmbeddings(tokens, tokenIdToEmbedding, entityEncoding)

    if (config.encodeWithSoftEntityLinking) {
      // Augment input embeddings with entity embeddings. We do this by adding
      // to each token embedding, a weighted average of the entity embeddings, weighed
      // by their similarity to that token. This is a soft entity linking.
      val entityEmbeddings = entityLinking.entities.map(e => encodeBow(e, tokenIdToEmbedding))
      val softEntityLinking = inputEmbeddings.map(rep => {
        val similarityValues = entityEmbeddings.map(entityVec => Expression.dotProduct(rep, entityVec))
        val invSimilaritySum = Expression.inverse(similarityValues.reduce(_ + _))
        similarityValues.zip(entityEmbeddings).map(sv => sv._1 * sv._2 * invSimilaritySum).reduce(_ + _)
      })
      // TODO: Try concatenation instead of sum?
      inputEmbeddings = inputEmbeddings.zip(softEntityLinking).map(vs => vs._1 + vs._2)
    }

    // TODO: should we add dropout to the builders using the .set_dropout methods?
    forwardBuilder.startNewSequence()
    val fwOutputs = ListBuffer[Expression]()
    for (inputEmbedding <- inputEmbeddings) {
      val fwOutput = forwardBuilder.addInput(inputEmbedding)
      val fwOutputDropout = if (dropoutProb > 0.0) {
        dropout(fwOutput, dropoutProb.asInstanceOf[Float])
      } else {
        fwOutput
      }
      fwOutputs += fwOutputDropout
    }

    backwardBuilder.startNewSequence()
    val bwOutputs = ListBuffer[Expression]()
    for (inputEmbedding <- inputEmbeddings.reverse) {
      val bwOutput = backwardBuilder.addInput(inputEmbedding)
      val bwOutputDropout = if (dropoutProb > 0.0) {
        dropout(bwOutput, dropoutProb.asInstanceOf[Float])
      } else {
        bwOutput
      }
      bwOutputs += bwOutputDropout
    }

    val outputEmbeddings = fwOutputs.toArray.zip(bwOutputs.toList.reverse).map(
        x => concatenateArray(Array(x._1, x._2)))

    val sentEmbedding = concatenateArray(Array(fwOutputs.last, bwOutputs.last))
    val inputMatrix = concatenateArray(inputEmbeddings.map(x => reshape(x, Dim(1, config.lstmInputDim))))
    val outputMatrix = concatenateArray(outputEmbeddings.map(reshape(_, Dim(1, 2 * config.hiddenDim))))

    // Initialize output lstm with the sum of the forward and backward
    // states.
    val forwardS = forwardBuilder.finalS
    val backwardS = backwardBuilder.finalS
    val s = if (config.concatLstmForDecoder) {
      new ExpressionVector(forwardS.zip(backwardS).map(
        x => Expression.concatenate(x._1, x._2)))
    } else {
      new ExpressionVector(forwardS.zip(backwardS).map(x => x._1 + x._2))
    }

    (s, sentEmbedding, inputMatrix, outputMatrix)
  }

  private def encodeEntities(entityLinking: EntityLinking,
                             tokens: Array[Int], tokenIdToEmbedding: Int => Expression): EntityEncoding = {

    val tokenEmbeddings = tokens.map(tokenIdToEmbedding(_))

    val typedEntityTokenMatrices = for {
      t <- entityLinking.entityTypes
    } yield {
      val entities = entityLinking.getEntitiesWithType(t)
      val entityLinkingFeaturizedParam = Expression.parameter(params(ENTITY_LINKING_FEATURIZED_PARAM + t))
      val entityLinkingBiasParam = Expression.parameter(params(ENTITY_LINKING_BIAS_PARAM + t))
      val mlpW1 = Expression.parameter(params(ENTITY_LINKING_FEATURIZED_MLP_W1 + t))
      val mlpB1 = Expression.parameter(params(ENTITY_LINKING_FEATURIZED_MLP_B1 + t))
      val mlpW2 = Expression.parameter(params(ENTITY_LINKING_FEATURIZED_MLP_W2 + t))

      val entityScores = entities.map(e => scoreEntityTokensMatch(e,
          tokens, tokenEmbeddings, tokenIdToEmbedding, entityLinking,
          entityLinkingFeaturizedParam, entityLinkingBiasParam, mlpW1, mlpB1, mlpW2))

      val entityScoreMatrix = Expression.concatenateCols(new ExpressionVector(entityScores))

      (t, entityScoreMatrix)
    }
    val knowledgeGraphEmbedding = embedKnowledgeGraph(entityLinking.graph, entityLinking.entities,
                                                      tokenIdToEmbedding)
    if (knowledgeGraphEmbedding.isEmpty) {
      println("WARNING: Knowledge graph embedding is empty!")
    }
    val entityEmbeddingMatrices = for {
      t <- entityLinking.entityTypes
    } yield {
      val entities = entityLinking.getEntitiesWithType(t)
      // val entityEmbeddings = entities.map(e => encodeBow(e, tokenIdToEmbedding))
      val entityEmbeddings = if (config.encodeEntitiesWithGraph) {
        entities.map(e => encodeEntityWithGraph(e, knowledgeGraphEmbedding))
      } else {
        entities.map(e => encodeType(e))
      }
      val entityEmbeddingMatrix = Expression.concatenateCols(new ExpressionVector(entityEmbeddings))

      (t, entityEmbeddingMatrix)
    }

    EntityEncoding(typedEntityTokenMatrices.toMap, entityEmbeddingMatrices.toMap, entityLinking)
  }

  private def scoreEntityTokensMatch(entity: Entity, tokens: Array[Int],
      tokenEmbeddings: Array[Expression], tokenIdToEmbedding: Int => Expression,
      entityLinking: EntityLinking, entityLinkingFeaturizedParam: Expression,
      entityLinkingFeaturizedBias: Expression, entityLinkingFeaturizedMlpW1: Expression,
      entityLinkingFeaturizedMlpB1: Expression, entityLinkingFeaturizedMlpW2: Expression): Expression = {

    val inVocabNameTokens = entity.nameTokensSet.filter(_ < vocab.size)
    val entityNameEmbeddings = inVocabNameTokens.map(tokenIdToEmbedding(_)).toArray

    var tokenScoresExpr = Expression.zeroes(Dim(tokens.length))

    if (config.entityLinkingLearnedSimilarity && entityNameEmbeddings.length >= 1) {
      if (config.maxPoolEntityTokenSimiliarities) {
        val scores = tokenEmbeddings.map{ t =>
          entityNameEmbeddings.map(e => Expression.dotProduct(t, e)).reduce((x, y) => Expression.max(x, y))
        }
        tokenScoresExpr = tokenScoresExpr + Expression.concatenate(new ExpressionVector(scores))
      } else {
        val entityEmbedding = entityNameEmbeddings.reduce((x, y) => Expression.sum(x, y))
        val scores = tokenEmbeddings.map(t => Expression.dotProduct(t, entityEmbedding))
        tokenScoresExpr = tokenScoresExpr + Expression.concatenate(new ExpressionVector(scores))
      }
    }

    if (config.featureGenerator.isDefined) {
      val (dim, floatVector) = entityLinking.getTokenFeatures(entity)
      val featureMatrix = Expression.input(dim, floatVector)
      val biasRepeated = Expression.concatenate(new ExpressionVector(
          List.fill(tokenEmbeddings.length)(entityLinkingFeaturizedBias)))

      if (config.featureMlp) {
        val layer1Bias = Expression.concatenate(new ExpressionVector(
          List.fill(tokenEmbeddings.length)(entityLinkingFeaturizedMlpB1)))
        val hidden = Expression.rectify((featureMatrix * entityLinkingFeaturizedMlpW1) + layer1Bias)

        tokenScoresExpr += (hidden * entityLinkingFeaturizedMlpW2) + biasRepeated
      } else {
        tokenScoresExpr += (featureMatrix * entityLinkingFeaturizedParam) + biasRepeated
      }
    }

    tokenScoresExpr
  }

  /*
  private def scoreEntityTokenMatch(entity: Entity, tokenId: Int,
      tokenEmbedding: Expression, tokenIndex: Int, entityNameEmbeddings: Array[Expression],
      entityLinkingFeaturizedParam: Expression): Expression = {
    var score = Expression.input(0.0f)

    if (config.entityLinkingLearnedSimilarity && tokenId < vocab.size) {
      if (entityNameEmbeddings.size > 0) {
        val entityTokenSimilarities = entityNameEmbeddings.map(x =>
          Expression.dotProduct(x, tokenEmbedding))

        if (entityTokenSimilarities.size > 1) {
          score = score + entityTokenSimilarities.reduce((x, y) => Expression.max(x, y))
        } else {
          score = score + entityTokenSimilarities(0)
        }
      }
    }
    score
  }
  */

  private def encodeBow(entity: Entity, tokenIdToEmbedding: Int => Expression): Expression = {
    if (entity.nameTokens.nonEmpty) {
      entity.nameTokens.map(tokenIdToEmbedding).reduce(_ + _) / entity.nameTokens.length
    } else {
      Expression.zeroes(Dim(config.inputDim))
    }
  }

  private def encodeType(entity: Entity): Expression = {
    // Expression.randomNormal(Dim(config.entityDim))
    // Expression.input(Dim(config.entityDim), new FloatVector(List.fill(config.entityDim)(0.0f)))
    val v = new FloatVector(List.fill(config.entityDim)(0.0f))
    v.update(actionSpace.typeIndex.getIndex(entity.t), 1.0f)
    Expression.input(Dim(config.entityDim), v)
  }

  def embedKnowledgeGraph(graph: KnowledgeGraph, entities: Array[Entity],
                          tokenIdToEmbeddings: Int => Expression): Map[Entity, Expression] = {
    val stringEntityMapping: Map[String, Entity] = graph.nodeSet.map(
      nodeName => nodeName -> entities.filter(e => e.expr.toString.equals(nodeName))(0)).toMap
    val graphEmbedding = for {
      nodeString <- graph.nodeSet
      node <- stringEntityMapping.get(nodeString)
      nodeNeighborEntities = graph.getNeighbors(nodeString).flatMap(n => stringEntityMapping.get(n))
      allNeighborReps = nodeNeighborEntities.map(e => encodeBow(e, tokenIdToEmbeddings))
    } yield {
      if (allNeighborReps.nonEmpty) {
        // TODO: Do something better than simple averaging.
        Some(node -> allNeighborReps.reduce(_ + _) / allNeighborReps.length)
      } else {
        None
      }
    }
    graphEmbedding.flatten.toMap
  }

  private def encodeEntityWithGraph(entity: Entity,
                                    graphEmbedding: Map[Entity, Expression]): Expression = {
    val zeroVector = Expression.zeroes(Dim(config.inputDim))
    val neighborRep = graphEmbedding.getOrElse(entity, zeroVector)
    val typeRep = encodeType(entity)
    val typeWeight = Expression.parameter(params(ENTITY_TYPE_INPUT_PARAM + entity.t))
    val typeBias = Expression.parameter(params(ENTITY_TYPE_INPUT_BIAS + entity.t))
    val neighborWeight = Expression.parameter(params(ENTITY_NEIGHBOR_INPUT_PARAM))
    val neighborBias = Expression.parameter(params(ENTITY_NEIGHBOR_INPUT_BIAS))
    val neighborProj = Expression.tanh(Expression.affineTransform(neighborBias, neighborWeight, neighborRep))
    val typeProj = Expression.tanh(Expression.affineTransform(typeBias, typeWeight, typeRep))
    neighborProj + typeProj
  }

  def generateLogProbs(wikiexample: WikiTablesExample):
  List[(Expression2, Expression)] = {

    val sentence = wikiexample.sentence
    val tokens = sentence.getAnnotation("tokenIds").asInstanceOf[Array[Int]]
    val entityLinking = sentence.getAnnotation("entityLinking").asInstanceOf[EntityLinking]
    val input = encode(tokens, entityLinking)
    // _ = println("parsing")

    // Choose the root type for the logical form given the
    // final output of the LSTM.
    val rootWeights = Expression.parameter(params(SemanticParserNoPnp.ROOT_WEIGHTS_PARAM))
    val rootBias = Expression.parameter(params(SemanticParserNoPnp.ROOT_BIAS_PARAM))
    val rootScores = Expression.logSoftmax((rootWeights * input.sentEmbedding) + rootBias)
    for {
      (rootType, lfTemplateSeqs) <- wikiexample.groupedTemplateSeqs
      rootScore = Expression.pick(rootScores, actionSpace.rootTypes.indexOf(rootType))
      state = SemanticParserState.start()
      (expr, logProb) <- parse(input, actionBuilder, state.addRootType(rootType), rootScore, lfTemplateSeqs.map(_._2))
    } yield {
      (expr.decodeExpression, logProb)
    }
  }

  private def parse(input: InputEncoding, builder: RnnBuilder,
      startState: SemanticParserState, lfScore: Expression, templates: List[List[Template]]):
  List[(SemanticParserState, Expression)] = {
    // Initialize the output LSTM before generating the logical form.
    builder.startNewSequence(input.rnnState)
    val startRnnState = builder.state()
    val beginActionsParam = Expression.parameter(params(SemanticParserNoPnp.BEGIN_ACTIONS +
      startState.unfilledHoleIds.head.t))
    parse(input, builder, beginActionsParam, startRnnState, startState, lfScore, templates, 0)
  }

  /** Recursively generates a logical form from a partial logical
    * form containing typed holes. Each application fills a single
    * hole with a template representing a constant, application or
    * lambda expression. These templates may contain additional holes
    * to be filled in the future. An LSTM is used to encode the history
    * of previously generated templates and select which template to
    * apply.
    */
  private def parse(input: InputEncoding, builder: RnnBuilder, prevInput: Expression,
      rnnState: Int, state: SemanticParserState, lfScore: Expression, templateSeqs: List[List[Template]], idx: Int):
  List[(SemanticParserState, Expression)] = {
    if (state.unfilledHoleIds.isEmpty) {
      // If there are no holes, return the completed logical form.
      templateSeqs.map(_ => (state, lfScore))
    } else {
      // Select the first unfilled hole and select the
      // applicable templates given the hole's type.
      val hole = state.unfilledHoleIds.head
      val actionTemplates = actionSpace.getTemplates(hole.t)
      val allVariableTemplates = hole.scope.getVariableTemplates(hole.t)
      val variableTemplates = if (allVariableTemplates.length > config.maxVars) {
        // The model only has parameters for MAX_VARS variables.
        allVariableTemplates.slice(0, config.maxVars)
      } else {
        allVariableTemplates
      }
      val baseTemplates = actionTemplates ++ variableTemplates

      val entities = input.entityEncoding.entityLinking.getEntitiesWithType(hole.t)
      val entityTemplates = entities.map(_.template)
      val entityTokenMatrix = input.entityEncoding.tokenEntityScoreMatrices.getOrElse(hole.t, null)

      val allTemplates = baseTemplates ++ entityTemplates

      // Update the LSTM and use its output to score
      // the applicable templates.
      // println("lstm add input")
      val rnnOutput = builder.addInput(rnnState, prevInput)
      val rnnOutputDropout = if (dropoutProb > 0.0) {
        Expression.dropout(rnnOutput, dropoutProb.asInstanceOf[Float])
      } else {
        rnnOutput
      }
      val nextRnnState = builder.state
      // println("computing attention")
        // Compute an attention vector
      val attentionWeights = Expression.parameter(params(SemanticParserNoPnp.ATTENTION_WEIGHTS_PARAM))
      val wordAttentions = Expression.transpose(
        Expression.softmax(input.encodedTokenMatrix * attentionWeights * rnnOutputDropout))

      // Attention vector using the input token vectors
      // attentionVector = transpose(wordAttentions * input.tokenMatrix)
      // Attention vector using the encoded tokens
      val attentionVector = Expression.transpose(wordAttentions * input.encodedTokenMatrix)

      // _ = println("action hidden weights")
      val actionHiddenWeights = Expression.parameter(params(SemanticParserNoPnp.ACTION_HIDDEN_WEIGHTS))
      val actionHiddenBias = Expression.parameter(params(SemanticParserNoPnp.ACTION_HIDDEN_BIAS))
      val actionHiddenWeights2 = Expression.parameter(params(SemanticParserNoPnp.ACTION_HIDDEN_ACTION + hole.t))
      val actionHiddenBias2 = Expression.parameter(params(SemanticParserNoPnp.ACTION_HIDDEN_ACTION_BIAS + hole.t))
      val attentionAndRnn = concatenateArray(Array(attentionVector, rnnOutputDropout))
      val hidden = if (config.actionBias) {
        (actionHiddenWeights * attentionAndRnn) + actionHiddenBias
      } else {
        actionHiddenWeights * attentionAndRnn
      }

      val actionHidden = if (config.relu) {
        Expression.rectify(hidden)
      } else {
        Expression.tanh(hidden)
      }

      val actionHiddenDropout = if (dropoutProb > 0.0) {
        Expression.dropout(actionHidden, dropoutProb.asInstanceOf[Float])
      } else {
        actionHidden
      }

      val actionHiddenScores = if (config.actionBias) {
        (actionHiddenWeights2 * actionHidden) + actionHiddenBias2
      } else {
        actionHiddenWeights2 * actionHidden
      }

      // Score the templates.
      val actionScores = Expression.pickrange(actionHiddenScores, 0, baseTemplates.length)

      // Score the entity templates
      // _ = println("scoring entities")
      val allScores = if (entities.length > 0) {
        val entityScores = wordAttentions * entityTokenMatrix
        val entityScoresVector = Expression.transpose(entityScores)
        concatenateArray(Array(actionScores, entityScoresVector))
      } else {
        actionScores
      }

      // Nondeterministically select which template to update
      // the parser's state with. The tag for this choice is
      // its index in the sequence of generated templates, which
      // can be used to supervise the parser.
      if (allTemplates.isEmpty) {
        println("Warning: no actions from hole " + hole)
      }
      val groups = templateSeqs.groupBy(_(idx)).toList
      for {
        (template, templates) <- groups
        templateTuple = (template, allTemplates.indexOf(template))
        nextLfScore = lfScore + Expression.pick(Expression.logSoftmax(allScores), templateTuple._2)
        nextState = templateTuple._1.apply(state).addAttention(wordAttentions)
        // Get the LSTM input parameters associated with the chosen
        // template.
        actionLookup = lookupParams(SemanticParserNoPnp.ACTION_LOOKUP_PARAM + hole.t)
        entityLookup = lookupParams(SemanticParserNoPnp.ENTITY_LOOKUP_PARAM + hole.t)
        index = templateTuple._2
        actionInput = if (index < baseTemplates.length) {
          Expression.lookup(actionLookup, templateTuple._2)
        } else {
          // TODO: using a single parameter vector for all entities of a given type
          // seems suboptimal.
          Expression.lookup(entityLookup, 0)
        }

        actionLstmInputWeights = Expression.parameter(params(SemanticParserNoPnp.ACTION_LSTM_INPUT_WEIGHTS))
        actionLstmInputBias = Expression.parameter(params(SemanticParserNoPnp.ACTION_LSTM_INPUT_BIAS))
        lstmInput1 = concatenateArray(Array(actionInput, attentionVector))
        lstmInput2 = if (config.actionLstmHiddenLayer) {
          val hidden = (actionLstmInputWeights * lstmInput1) + actionLstmInputBias
          if (config.relu) {
            Expression.rectify(hidden)
          } else {
            Expression.tanh(hidden)
          }
        } else {
          lstmInput1
        }

        // _ = println("recursing")
        // Recursively fill in any remaining holes.
        (returnState, totalLfScore) <- parse(input, builder, lstmInput2, nextRnnState,
          nextState, nextLfScore, templates, idx + 1)
      } yield {
        (returnState, totalLfScore)
      }
    }
  }

  /** Generate a distribution over logical forms given
    * tokens.
    */
  def parse(tokens: Array[Int], entityLinking: EntityLinking): Pnp[SemanticParserState] = {
    val input = encode(tokens, entityLinking)

    // _ = println("parsing")
    val state = SemanticParserState.start

    // Choose the root type for the logical form given the
    // final output of the LSTM.
    val rootWeights = Expression.parameter(params(SemanticParserNoPnp.ROOT_WEIGHTS_PARAM))
    val rootBias = Expression.parameter(params(SemanticParserNoPnp.ROOT_BIAS_PARAM))
    val rootScores = Expression.logSoftmax((rootWeights * input.sentEmbedding) + rootBias)
    for {
      // Encode input tokens using an LSTM.
      rootType <- Pnp.choose(actionSpace.rootTypes, rootScores, state)

      // _ = println("parsing 2")
      // Recursively generate a logical form using an LSTM to
      // select logical form templates to expand on typed holes
      // in the partially-generated logical form.
      expr <- parse(input, actionBuilder, state.addRootType(rootType))
    } yield {
      expr
    }
  }

  private def parse(input: InputEncoding, builder: RnnBuilder,
                    startState: SemanticParserState): Pnp[SemanticParserState] = {
    // Initialize the output LSTM before generating the logical form.
    builder.startNewSequence(input.rnnState)
    val startRnnState = builder.state()
    val beginActionsParam = Expression.parameter(params(SemanticParserNoPnp.BEGIN_ACTIONS +
      startState.unfilledHoleIds.head.t))
    for {
      e <- parse(input, builder, beginActionsParam, startRnnState, startState)
    } yield {
      e
    }
  }

  /** Recursively generates a logical form from a partial logical
    * form containing typed holes. Each application fills a single
    * hole with a template representing a constant, application or
    * lambda expression. These templates may contain additional holes
    * to be filled in the future. An LSTM is used to encode the history
    * of previously generated templates and select which template to
    * apply.
    */
  private def parse(input: InputEncoding, builder: RnnBuilder, prevInput: Expression,
                    rnnState: Int, state: SemanticParserState): Pnp[SemanticParserState] = {
    if (state.unfilledHoleIds.isEmpty) {
      // If there are no holes, return the completed logical form.
      Pnp.value(state)
    } else {
      // Select the first unfilled hole and select the
      // applicable templates given the hole's type.
      val hole = state.unfilledHoleIds.head
      val actionTemplates = actionSpace.getTemplates(hole.t)
      val allVariableTemplates = hole.scope.getVariableTemplates(hole.t)
      val variableTemplates = if (allVariableTemplates.length > config.maxVars) {
        // The model only has parameters for MAX_VARS variables.
        allVariableTemplates.slice(0, config.maxVars)
      } else {
        allVariableTemplates
      }
      val baseTemplates = actionTemplates ++ variableTemplates

      val entities = input.entityEncoding.entityLinking.getEntitiesWithType(hole.t)
      val entityTemplates = entities.map(_.template)
      val entityTokenMatrix = input.entityEncoding.tokenEntityScoreMatrices.getOrElse(hole.t, null)

      val allTemplates = baseTemplates ++ entityTemplates

      // Update the LSTM and use its output to score
      // the applicable templates.
      // println("lstm add input")
      val rnnOutput = builder.addInput(rnnState, prevInput)
      val rnnOutputDropout = if (dropoutProb > 0.0) {
        Expression.dropout(rnnOutput, dropoutProb.asInstanceOf[Float])
      } else {
        rnnOutput
      }
      val nextRnnState = builder.state
      // println("computing attention")
      // Compute an attention vector
      val attentionWeights = Expression.parameter(params(SemanticParserNoPnp.ATTENTION_WEIGHTS_PARAM))
      val wordAttentions = Expression.transpose(
        Expression.softmax(input.encodedTokenMatrix * attentionWeights * rnnOutputDropout))

      // Attention vector using the input token vectors
      // attentionVector = transpose(wordAttentions * input.tokenMatrix)
      // Attention vector using the encoded tokens
      val attentionVector = Expression.transpose(wordAttentions * input.encodedTokenMatrix)

      // _ = println("action hidden weights")
      val actionHiddenWeights = Expression.parameter(params(SemanticParserNoPnp.ACTION_HIDDEN_WEIGHTS))
      val actionHiddenBias = Expression.parameter(params(SemanticParserNoPnp.ACTION_HIDDEN_BIAS))
      val actionHiddenWeights2 = Expression.parameter(params(SemanticParserNoPnp.ACTION_HIDDEN_ACTION + hole.t))
      val actionHiddenBias2 = Expression.parameter(params(SemanticParserNoPnp.ACTION_HIDDEN_ACTION_BIAS + hole.t))
      val attentionAndRnn = concatenateArray(Array(attentionVector, rnnOutputDropout))
      val hidden = if (config.actionBias) {
        (actionHiddenWeights * attentionAndRnn) + actionHiddenBias
      } else {
        actionHiddenWeights * attentionAndRnn
      }

      val actionHidden = if (config.relu) {
        Expression.rectify(hidden)
      } else {
        Expression.tanh(hidden)
      }

      val actionHiddenDropout = if (dropoutProb > 0.0) {
        Expression.dropout(actionHidden, dropoutProb.asInstanceOf[Float])
      } else {
        actionHidden
      }

      val actionHiddenScores = if (config.actionBias) {
        (actionHiddenWeights2 * actionHidden) + actionHiddenBias2
      } else {
        actionHiddenWeights2 * actionHidden
      }

      // Score the templates.
      val actionScores = Expression.pickrange(actionHiddenScores, 0, baseTemplates.length)

      // Score the entity templates
      // _ = println("scoring entities")
      val allScores = if (entities.length > 0) {
        val entityScores = wordAttentions * entityTokenMatrix
        val entityScoresVector = Expression.transpose(entityScores)
        concatenateArray(Array(actionScores, entityScoresVector))
      } else {
        actionScores
      }

      // Nondeterministically select which template to update
      // the parser's state with. The tag for this choice is
      // its index in the sequence of generated templates, which
      // can be used to supervise the parser.
      if (allTemplates.isEmpty) {
        println("Warning: no actions from hole " + hole)
      }
      for {
        templateTuple <- Pnp.choose(allTemplates.zipWithIndex.toArray, allScores, state)
        nextState = templateTuple._1.apply(state).addAttention(wordAttentions)

        // Get the LSTM input parameters associated with the chosen
        // template.
        actionLookup = lookupParams(SemanticParserNoPnp.ACTION_LOOKUP_PARAM + hole.t)
        entityLookup = lookupParams(SemanticParserNoPnp.ENTITY_LOOKUP_PARAM + hole.t)
        index = templateTuple._2
        actionInput = if (index < baseTemplates.length) {
          Expression.lookup(actionLookup, templateTuple._2)
        } else {
          // TODO: using a single parameter vector for all entities of a given type
          // seems suboptimal.
          Expression.lookup(entityLookup, 0)
        }

        actionLstmInputWeights = Expression.parameter(params(SemanticParserNoPnp.ACTION_LSTM_INPUT_WEIGHTS))
        actionLstmInputBias = Expression.parameter(params(SemanticParserNoPnp.ACTION_LSTM_INPUT_BIAS))
        lstmInput1 = concatenateArray(Array(actionInput, attentionVector))
        lstmInput2 = if (config.actionLstmHiddenLayer) {
          val hidden = (actionLstmInputWeights * lstmInput1) + actionLstmInputBias
          if (config.relu) {
            Expression.rectify(hidden)
          } else {
            Expression.tanh(hidden)
          }
        } else {
          lstmInput1
        }

        // _ = println("recursing")
        // Recursively fill in any remaining holes.
        returnState <- parse(input, builder, lstmInput2, nextRnnState, nextState)
      } yield {
        returnState
      }
    }
  }

  private def concatenateArray(exprs: Array[Expression]): Expression = {
    val expressionVector = new ExpressionVector(exprs.length)
    for (i <- exprs.indices) {
      expressionVector.update(i, exprs(i))
    }
    Expression.concatenate(expressionVector)
  }

  /** Generate the sequence of parser actions that produces exp.
    * This method assumes that only one such action sequence exists.
    */
  def generateActionSequence(exp: Expression2, entityLinking: EntityLinking,
      typeDeclaration: TypeDeclaration): Option[(List[Type], List[Template])] = {
    val holeIndexMap = MutableMap[Int, Int]()
    val actionTypes = ListBuffer[Type]()
    val actions = ListBuffer[Template]()

    val typeMap = StaticAnalysis.inferTypeMap(exp, TypeDeclaration.TOP, typeDeclaration)
      .asScala.toMap

    var state = SemanticParserState.start().addRootType(typeMap(0))
    holeIndexMap(state.nextHole().get.id) = 0

    while (state.nextHole().isDefined) {
      val hole = state.nextHole().get
      val expIndex = holeIndexMap(hole.id)
      val currentScope = hole.scope

      val curType = typeMap(expIndex)
      Preconditions.checkState(curType.equals(hole.t),
          "type-checked type %s does not match hole %s at index %s of %s",
          curType, hole.t, expIndex.asInstanceOf[Integer], exp)

      val templates = actionSpace.getTemplates(curType) ++
        currentScope.getVariableTemplates(curType) ++
        entityLinking.getEntitiesWithType(curType).map(_.template)
      val matches = templates.filter(_.matches(expIndex, exp, typeMap))
      if (matches.size != 1) {
        println("Found " + matches.size + " for expression " + exp.getSubexpression(expIndex) +
            " : "  + curType + " (expected 1)")
        return None
      }

      val theMatch = matches.toList.head
      state = theMatch.apply(state)

      actionTypes += curType
      actions += theMatch

      var holeOffset = 0
      for ((holeIndex, hole) <- theMatch.holeIndexes.zip(
          state.unfilledHoleIds.slice(0, theMatch.holeIndexes.length))) {
        val curIndex = expIndex + holeIndex + holeOffset
        holeIndexMap(hole.id) = curIndex
        holeOffset += (exp.getSubexpression(curIndex).size - 1)
      }
    }

    val decoded = state.decodeExpression
    Preconditions.checkState(decoded.equals(exp), "Expected %s and %s to be equal", decoded, exp)

    Some((actionTypes.toList, actions.toList))
  }

  /** Generate an execution score that constrains the
    * parser to generate exp. This oracle can be used
    * to supervise the parser when training with loglikelihood.
    */
  def getLabelScore(exp: Expression2, entityLinking: EntityLinking,
      typeDeclaration: TypeDeclaration): Option[SemanticParserExecutionScore] = {
    for {
      (holeTypes, templates) <- generateActionSequence(exp, entityLinking, typeDeclaration)
    } yield {
      new SemanticParserExecutionScore(holeTypes.toArray, templates.toArray,
          Double.NegativeInfinity)
    }
  }

  def getMarginScore(exp: Expression2, entityLinking: EntityLinking,
      typeDeclaration: TypeDeclaration): Option[SemanticParserExecutionScore] = {
    for {
      (holeTypes, templates) <- generateActionSequence(exp, entityLinking, typeDeclaration)
    } yield {
      new SemanticParserExecutionScore(holeTypes.toArray, templates.toArray, 1.0)
    }
  }

  /**
   * Generate an execution score that constrains the parser
   * to produce any expression in {@code exprs}. This oracle
   * can be used to supervise the parser when training with
   * loglikelihood with weak supervision, i.e., when multiple
   * logical forms may be correct.
   */
  def getMultiLabelScore(exprs: Iterable[Expression2], entityLinking: EntityLinking,
      typeDeclaration: TypeDeclaration): Option[SemanticParserMultiExecutionScore] = {
    getMultiScore(exprs, entityLinking, typeDeclaration, Double.NegativeInfinity)
  }

  def getMultiMarginScore(exprs: Iterable[Expression2], entityLinking: EntityLinking,
      typeDeclaration: TypeDeclaration): Option[SemanticParserMultiExecutionScore] = {
    getMultiScore(exprs, entityLinking, typeDeclaration, 1.0)
  }

  private def getMultiScore(exprs: Iterable[Expression2], entityLinking: EntityLinking,
      typeDeclaration: TypeDeclaration, incorrectCost: Double
      ): Option[SemanticParserMultiExecutionScore] = {
    val oracles = for {
      lf <- exprs
      oracle <- getLabelScore(lf, entityLinking, typeDeclaration)
    } yield {
      oracle
    }

    if (oracles.nonEmpty) {
      val score = new SemanticParserMultiExecutionScore(oracles, incorrectCost)
      Some(score)
    } else {
      None
    }
  }


  /**
   * Serialize this parser into {@code saver}. This method
   * assumes that the {@code model} has already been
   * serialized (using {@code model.save}).
   */
  def save(saver: ModelSaver): Unit = {
    saver.addModel(model)
    saver.addInt(params.size)
    for ((k, v) <- params) {
      saver.addObject(k)
      saver.addParameter(v)
    }

    saver.addInt(lookupParams.size)
    for ((k, v) <- lookupParams) {
      saver.addObject(k)
      saver.addLookupParameter(v)
    }
    saver.addObject(actionSpace)
    saver.addObject(vocab)
    saver.addObject(config)
    saver.addLstmBuilder(forwardBuilder)
    saver.addLstmBuilder(backwardBuilder)
    saver.addLstmBuilder(actionBuilder)
  }
}

///** Execution score that constrains a SemanticParserNoPnp
//  * to generate the given rootType and sequence of
//  * templates.
//  */
//class SemanticParserExecutionScore(val holeTypes: Array[Type],
//    val templates: Array[Template], val incorrectCost: Double)
//extends ExecutionScore {
//
//  val rootType = holeTypes(0)
//  val reversedTemplates = templates.reverse
//
//  def apply(tag: Any, choice: Any, env: Env): Double = {
//    if (tag != null && tag.isInstanceOf[SemanticParserState]) {
//      val state = tag.asInstanceOf[SemanticParserState]
//      if (state.numActions == 0 && state.unfilledHoleIds.isEmpty) {
//        // This state represents the start of parsing where we
//        // choose the root type.
//        Preconditions.checkArgument(choice.isInstanceOf[Type])
//        if (choice.asInstanceOf[Type].equals(rootType)) {
//          0.0
//        } else {
//          incorrectCost
//        }
//      } else {
//        val actionInd = state.numActions
//        if (actionInd < templates.length) {
//          // TODO: this test may be too inefficient.
//          val chosen = choice.asInstanceOf[(Template, Int)]._1
//          val myTemplates = reversedTemplates.slice(reversedTemplates.length - actionInd, reversedTemplates.length)
//          if (chosen.equals(templates(actionInd)) &&
//              (actionInd == 0 || state.templates.zip(myTemplates).map(x => x._1.equals(x._2)).reduce(_ && _))) {
//            0.0
//          } else {
//            incorrectCost
//          }
//        } else {
//          incorrectCost
//        }
//      }
//    } else {
//      0.0
//    }
//  }
//}
//
//class MaxExecutionScore(val scores: Seq[ExecutionScore]) extends ExecutionScore {
//  def apply(tag: Any, choice: Any, env: Env): Double = {
//    return scores.map(s => s.apply(tag, choice, env)).max
//  }
//}
//
//class SemanticParserMultiExecutionScore(val scores: Iterable[SemanticParserExecutionScore],
//    val incorrectCost: Double)
//  extends ExecutionScore {
//
//  val trie = new Trie[AnyRef]()
//
//  for (score <- scores) {
//    val key = List(score.rootType) ++ score.templates
//    trie.insert(key)
//  }
//
//  def apply(tag: Any, choice: Any, env: Env): Double = {
//    if (tag != null && tag.isInstanceOf[SemanticParserState]) {
//      val state = tag.asInstanceOf[SemanticParserState]
//      val key = if (state.hasRootType) {
//        Array(state.rootType) ++ state.getTemplates
//      } else {
//        Array()
//      }
//
//      val trieNodeId = trie.lookup(key)
//      if (trieNodeId.isDefined) {
//        val nextStates = trie.getNextMap(trieNodeId.get)
//
//        // First action is a type. Remaining actions are templates, but they
//        // come paired with an index that the parser uses for selection purposes.
//        val action: AnyRef = if (state.hasRootType) {
//          choice.asInstanceOf[(Template, Int)]._1
//        } else {
//          choice.asInstanceOf[Type]
//        }
//
//        if (nextStates.contains(action)) {
//          0.0
//        } else {
//          incorrectCost
//        }
//      } else {
//        incorrectCost
//      }
//    } else {
//      0.0
//    }
//  }
//}
//
//class SemanticParserConfig extends Serializable {
//
//  var inputDim = 200
//  var hiddenDim = 100
//  var actionDim = 100
//  var actionHiddenDim = 100
//  var maxVars = 10
//
//  var entityDim = -1
//  def lstmInputDim = inputDim + entityDim
//
//  var relu = false
//  var actionBias = false
//  var actionLstmHiddenLayer = false
//  var concatLstmForDecoder = false
//
//  var featureGenerator: Option[SemanticParserFeatureGenerator] = None
//  var featureMlp = false
//  var featureMlpDim = 50
//
//  var entityLinkingLearnedSimilarity = false
//  var maxPoolEntityTokenSimiliarities = false
//  var encodeWithSoftEntityLinking = false
//  var encodeEntitiesWithGraph = false
//  var distinctUnkVectors = false
//
//  // XXX: I'm not sure if these really belong here, but we want to serialize
//  // them with the parser.
//  var preprocessor: LfPreprocessor = null
//  var typeDeclaration: TypeDeclaration = null
//}

object SemanticParserNoPnp {

  // Parameter names used by the parser
  val WORD_EMBEDDINGS_PARAM = "wordEmbeddings"
  val ROOT_WEIGHTS_PARAM = "rootWeights"
  val ROOT_BIAS_PARAM = "rootBias"

  val BEGIN_ACTIONS = "beginActions:"
  val ACTION_LOOKUP_PARAM = "actionLookup:"

  val ATTENTION_WEIGHTS_PARAM = "attentionWeights:"

  val ACTION_HIDDEN_WEIGHTS = "actionHidden"
  val ACTION_HIDDEN_BIAS = "actionHiddenBias"
  val ACTION_HIDDEN_ACTION = "actionHiddenOutput:"
  val ACTION_HIDDEN_ACTION_BIAS = "actionHiddenOutputBias:"

  val ACTION_LSTM_INPUT_WEIGHTS = "actionLstmInputWeights"
  val ACTION_LSTM_INPUT_BIAS = "actionLstmInputBias"

  val ENTITY_BIAS_PARAM = "entityBias:"
  val ENTITY_WEIGHTS_PARAM = "entityWeights:"
  val ENTITY_LOOKUP_PARAM = "entityLookup:"

  val ENTITY_LINKING_FEATURIZED_PARAM = "entityLinkingFeaturized:"
  val ENTITY_LINKING_BIAS_PARAM = "entityLinkingBias:"
  val ENTITY_LINKING_FEATURIZED_MLP_W1 = "entityLinkingFeaturizedMlpW1:"
  val ENTITY_LINKING_FEATURIZED_MLP_B1 = "entityLinkingFeaturizedMlpB1:"
  val ENTITY_LINKING_FEATURIZED_MLP_W2 = "entityLinkingFeaturizedMlpW2:"

  val ENTITY_TYPE_INPUT_PARAM = "entityTypeInput:"
  val ENTITY_TYPE_INPUT_BIAS = "entityTypeBias:"
  val ENTITY_NEIGHBOR_INPUT_PARAM = "entityTypeInput:"
  val ENTITY_NEIGHBOR_INPUT_BIAS = "entityTypeBias:"

  //TODO: make these instance members
  var params: Map[String, Parameter] = Map()
  var lookupParams: Map[String, LookupParameter] = Map[String, LookupParameter]()
  var model = new Model()

  def addParameter(name: String, dim: Dim) = {
    params += (name -> model.addParameters(dim))
  }

  def addParameter(name: String, dim: Dim, init: ParameterInit) = {
    params += (name -> model.addParameters(dim, init))
  }

  def addLookupParameter(name: String, lookupNum: Long, dim: Dim) = {
    lookupParams += (name -> model.addLookupParameters(lookupNum, dim))
  }

  def addLookupParameter(name: String, lookupNum: Long, dim: Dim,
                         init: ParameterInit) = {
    lookupParams += (name -> model.addLookupParameters(lookupNum, dim, init))
  }

  def getParameter(name: String): Parameter = {
    params(name)
  }

  def getLookupParameter(name: String): LookupParameter = {
    lookupParams(name)
  }

  def create(actionSpace: ActionSpace, vocab: IndexedList[String],
      wordEmbeddings: Option[Array[(String, FloatVector)]],
      config: SemanticParserConfig): SemanticParserNoPnp = {
    // XXX: fix this
    config.entityDim = actionSpace.typeIndex.size()
    val actionLstmHiddenDim = if (config.concatLstmForDecoder) {
      config.hiddenDim * 2
    } else {
      config.hiddenDim
    }
    val actionLstmInputDim = config.actionDim + 2 * config.hiddenDim

    // Initialize model
    // TODO: document these parameters.
    addParameter(ROOT_WEIGHTS_PARAM, Dim(actionSpace.rootTypes.length, 2 * config.hiddenDim))
    addParameter(ROOT_BIAS_PARAM, Dim(actionSpace.rootTypes.length))
    addParameter(ATTENTION_WEIGHTS_PARAM, Dim(2 * config.hiddenDim, actionLstmHiddenDim))

    addParameter(ACTION_HIDDEN_WEIGHTS, Dim(config.actionHiddenDim,
        2 * config.hiddenDim + actionLstmHiddenDim))
    addParameter(ACTION_HIDDEN_BIAS, Dim(config.actionHiddenDim))

    addParameter(ACTION_LSTM_INPUT_WEIGHTS, Dim(actionLstmInputDim, actionLstmInputDim))
    addParameter(ACTION_LSTM_INPUT_BIAS, Dim(actionLstmInputDim))

    // The last entry will be the unknown word.
    if (wordEmbeddings.isDefined) {
      val embeddings = wordEmbeddings.get
      val embeddingDim = embeddings(0)._2.length
      val initializerSize = embeddings.length * embeddingDim

      // values are stored in column-major format
      val initializer = new FloatVector(initializerSize)
      for (i <- embeddings.indices) {
        for (j <- 0 until embeddingDim) {
          val idx = i * embeddingDim + j
          initializer.update(idx, embeddings(i)._2(j))
        }
      }

      val parameterInit = ParameterInit.fromVector(initializer)
      addLookupParameter(WORD_EMBEDDINGS_PARAM, vocab.size + 1, Dim(config.inputDim), parameterInit)
    } else {
      addLookupParameter(WORD_EMBEDDINGS_PARAM, vocab.size + 1, Dim(config.inputDim))
    }

    for (t <- actionSpace.getTypes()) {
      val actions = actionSpace.getTemplates(t)
      val dim = actions.length + config.maxVars

      addParameter(BEGIN_ACTIONS + t, Dim(config.actionDim + 2 * config.hiddenDim))
      addLookupParameter(ACTION_LOOKUP_PARAM + t, dim, Dim(config.actionDim))
      addParameter(ACTION_HIDDEN_ACTION + t, Dim(dim, config.actionHiddenDim))
      addParameter(ACTION_HIDDEN_ACTION_BIAS + t, Dim(dim))

      addLookupParameter(ENTITY_LOOKUP_PARAM + t, 1, Dim(config.actionDim))

      if (config.featureGenerator.isDefined) {
        addParameter(ENTITY_LINKING_FEATURIZED_PARAM + t,
            Dim(config.featureGenerator.get.numFeatures))
        addParameter(ENTITY_LINKING_BIAS_PARAM + t, Dim(1))

        addParameter(ENTITY_LINKING_FEATURIZED_MLP_W1 + t,
            Dim(config.featureGenerator.get.numFeatures, config.featureMlpDim))
        addParameter(ENTITY_LINKING_FEATURIZED_MLP_B1 + t, Dim(1, config.featureMlpDim))
        addParameter(ENTITY_LINKING_FEATURIZED_MLP_W2 + t, Dim(config.featureMlpDim))
      }
      addParameter(ENTITY_TYPE_INPUT_PARAM + t, Dim(config.entityDim, config.entityDim))
      addParameter(ENTITY_TYPE_INPUT_BIAS + t, Dim(config.entityDim))
    }

    if (config.encodeEntitiesWithGraph) {
      addParameter(ENTITY_NEIGHBOR_INPUT_PARAM, Dim(config.entityDim, config.inputDim))
      addParameter(ENTITY_NEIGHBOR_INPUT_BIAS, Dim(config.entityDim))
    }

    // Forward and backward RNNs for encoding the input token sequence
    val forwardBuilder = new LstmBuilder(1, config.lstmInputDim, config.hiddenDim, model)
    val backwardBuilder = new LstmBuilder(1, config.lstmInputDim, config.hiddenDim, model)
    // RNN for generating actions given previous actions (and the input)
    val actionBuilder = new LstmBuilder(1, actionLstmInputDim,
        actionLstmHiddenDim, model)

    new SemanticParserNoPnp(actionSpace, vocab, config, forwardBuilder,
        backwardBuilder, actionBuilder, model, params, lookupParams)
  }

  def load(loader: ModelLoader): SemanticParserNoPnp = {
    val model = loader.loadModel()
    val numParams = loader.loadInt()
    val params = ListBuffer[(String, Parameter)]()
    for (i <- 0 until numParams) {
      val name = loader.loadObject(classOf[String])
      val param = loader.loadParameter()
      params += ((name, param))
    }

    val numLookups = loader.loadInt()
    val lookups = ListBuffer[(String, LookupParameter)]()
    for (i <- 0 until numLookups) {
      val name = loader.loadObject(classOf[String])
      val param = loader.loadLookupParameter()
      lookups += ((name, param))
    }
    val actionSpace = loader.loadObject(classOf[ActionSpace])
    val vocab = loader.loadObject(classOf[IndexedList[String]])
    val config = loader.loadObject(classOf[SemanticParserConfig])
    val forwardBuilder = loader.loadLstmBuilder()
    val backwardBuilder = loader.loadLstmBuilder()
    val actionBuilder = loader.loadLstmBuilder()
    new SemanticParserNoPnp(actionSpace, vocab, config, forwardBuilder,
        backwardBuilder, actionBuilder, model, params.toMap, lookups.toMap)
  }

  // TODO: move this method somewhere else.
  def seqToMultimap[A, B](s: Iterable[(A, B)]) = {
    s.foldLeft(new mutable.HashMap[A, MutableSet[B]] with mutable.MultiMap[A, B]){
      (acc, pair) => acc.addBinding(pair._1, pair._2)
    }
  }
}

///**
// * The row dimension of tokenMatrix, encodedTokenMatrix and
// * tokenEntityScoreMatrices corresponds to tokens.
// */
//case class InputEncoding(val tokens: Array[Int], val rnnState: ExpressionVector,
//    val sentEmbedding: Expression, val tokenMatrix: Expression, val encodedTokenMatrix: Expression,
//    val entityEncoding: EntityEncoding) {
//}
//
///**
// * The columns of tokenEntityScoreMatrices and entityEmbeddingMatrices
// * correspond to entities of that type, in the same order as in
// * entityLinking.
// */
//case class EntityEncoding(val tokenEntityScoreMatrices: Map[Type, Expression],
//    val entityEmbeddingMatrices: Map[Type, Expression], val entityLinking: EntityLinking)
//
//class ExpressionDecodingException extends RuntimeException {
//
//}
