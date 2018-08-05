package org.allenai.wikitables

import scala.collection.JavaConverters._
import scala.collection.mutable.{Map => MutableMap}
import org.allenai.pnp.Env
import org.allenai.pnp.PnpInferenceContext
import org.allenai.pnp.PnpModel
import org.allenai.pnp.semparse.EntityLinking
import org.allenai.pnp.semparse.SemanticParser
import org.allenai.pnp.semparse.SemanticParserLoss
import org.allenai.pnp.semparse.SemanticParserState
import com.jayantkrish.jklol.ccg.lambda.ExpressionParser
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.ExpressionComparator
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.SimplificationComparator
import com.jayantkrish.jklol.cli.AbstractCli
import edu.cmu.dynet._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec

import scala.collection.mutable.ListBuffer
import edu.stanford.nlp.sempre._
import java.nio.file.Paths
import java.nio.file.Files
import java.nio.charset.StandardCharsets
import java.io.PrintWriter
import java.io.File

import com.jayantkrish.jklol.util.CountAccumulator
import com.jayantkrish.jklol.ccg.lambda2.Expression2
import edu.stanford.nlp.sempre.tables.TableKnowledgeGraph
import fig.basic.LispTree

class TestWikiTablesCli extends AbstractCli() {
  
  import TestWikiTablesCli._
  
  var randomSeedOpt: OptionSpec[Long] = null
  var testDataOpt: OptionSpec[String] = null
  var derivationsPathOpt: OptionSpec[String] = null
  var noDerivationsOpt: OptionSpec[Void] = null
  var modelOpt: OptionSpec[String] = null

  var tsvOutputOpt: OptionSpec[String] = null
  var scoresOutputOpt: OptionSpec[String] = null
  
  var beamSizeOpt: OptionSpec[Integer] = null
  var evaluateDpdOpt: OptionSpec[Void] = null
  var maxDerivationsOpt: OptionSpec[Integer] = null

  // Options for answering single questions.
  var questionOpt: OptionSpec[String] = null
  var tableStringOpt: OptionSpec[String] = null
  var numAnswersOpt: OptionSpec[Integer] = null

  override def initializeOptions(parser: OptionParser): Unit = {
    //randomSeedOpt = parser.accepts("randomSeed").withRequiredArg().ofType(classOf[Long]).defaultsTo(2732932987L)
    randomSeedOpt = parser.accepts("randomSeed").withRequiredArg().ofType(classOf[Long]).defaultsTo(3287527509L)
    testDataOpt = parser.accepts("testData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',')
    derivationsPathOpt = parser.accepts("derivationsPath").withRequiredArg().ofType(classOf[String])
    noDerivationsOpt = parser.accepts("noDerivations")
    modelOpt = parser.accepts("model").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    
    tsvOutputOpt = parser.accepts("tsvOutput").withRequiredArg().ofType(classOf[String])

    beamSizeOpt = parser.accepts("beamSize").withRequiredArg().ofType(classOf[Integer]).defaultsTo(5)
    evaluateDpdOpt = parser.accepts("evaluateDpd")
    maxDerivationsOpt = parser.accepts("maxDerivations").withRequiredArg().ofType(classOf[Integer]).defaultsTo(-1)

    questionOpt = parser.accepts("question").withRequiredArg().ofType(classOf[String])
    tableStringOpt = parser.accepts("tableString").withRequiredArg().ofType(classOf[String])
    numAnswersOpt = parser.accepts("numAnswers").withRequiredArg().ofType(classOf[Integer])

    scoresOutputOpt = parser.accepts("scoresOutput").withRequiredArg().ofType(classOf[String])
  }

  override def run(options: OptionSet): Unit = {
    Initialize.initialize(Map("dynet-mem" -> "4096", "random-seed" -> options.valueOf(randomSeedOpt)))

    // Get the predicted denotations of each model. (and print out 
    // error analysis)
    val modelDenotations = if (options.has(questionOpt)) {
      // Single question mode
      options.valuesOf(modelOpt).asScala.map { modelFilename =>
        answerSingleQuestion(modelFilename, options)
      }
    } else {
      options.valuesOf(modelOpt).asScala.map { modelFilename =>
        runTestFile(modelFilename, options)
      }
    }


    val denotations = if (modelDenotations.size > 1) {
      val exIds = modelDenotations.head.keySet
      val denotationMap = MutableMap[String, List[(Value, Double)]]()
      for (exId <- exIds) {
        val valueScores = modelDenotations.flatMap(_(exId))
        val accumulator = CountAccumulator.create[Value]
        valueScores.foreach(x => accumulator.increment(x._1, x._2))
        val sortedValues = accumulator.getSortedKeys.asScala.map(x => (x, accumulator.getCount(x)))
        denotationMap(exId) = sortedValues.toList
      }
      denotationMap.toMap
    } else {
      modelDenotations.head
    }
    
    /*
    println("*** Validating test set action space ***")
    val testSeparatedLfs = WikiTablesSemanticParserCli.getCcgDataset(testPreprocessed)
    SemanticParserUtils.validateActionSpace(testSeparatedLfs, parser, typeDeclaration)
    */
    
    if (options.has(tsvOutputOpt)) {
      val filename = options.valueOf(tsvOutputOpt)
      val tsvStrings = denotations.map { d =>
        d._1 + "\t" + getAnswerTsvParts(d._2.map(_._1)).mkString("\t")
      }

      Files.write(Paths.get(filename), tsvStrings.mkString("\n").getBytes(StandardCharsets.UTF_8))
    }
  }

  def runTestFile(modelFilename: String, options: OptionSet): Map[String, List[(Value, Double)]] = {
    val simplifier = ExpressionSimplifier.lambdaCalculus()
    val comparator = new SimplificationComparator(simplifier)

    val parser = loadSerializedParser(modelFilename)
    val featureGenerator = parser.config.featureGenerator.get
    val typeDeclaration = parser.config.typeDeclaration
    val lfPreprocessor = parser.config.preprocessor

    // Read test data.
    val testData = WikiTablesUtil.loadDatasets(options.valuesOf(testDataOpt).asScala,
        options.valueOf(derivationsPathOpt), options.valueOf(maxDerivationsOpt),
        lfPreprocessor, !options.has(noDerivationsOpt))
    println("Read " + testData.size + " test examples")

    testData.foreach(x => WikiTablesUtil.preprocessExample(x, parser.vocab,
        featureGenerator, typeDeclaration))

    val (testResults, denotations) = TestWikiTablesCli.test(testData.map(_.ex),
        parser, options.valueOf(beamSizeOpt), options.has(evaluateDpdOpt),
        true, typeDeclaration, comparator, lfPreprocessor, println, options.valueOf(scoresOutputOpt))
    println("*** Evaluation results ***")
    println(testResults)

    denotations
  }

  def answerSingleQuestion(modelFilename: String, options: OptionSet): Map[String, List[(Value, Double)]] = {
    // TODO: Do not load model for each question. Different Cli?
    val simplifier = ExpressionSimplifier.lambdaCalculus()
    val comparator = new SimplificationComparator(simplifier)
    val parser = loadSerializedParser(modelFilename)
    val featureGenerator = parser.config.featureGenerator.get
    val typeDeclaration = parser.config.typeDeclaration
    val lfPreprocessor = parser.config.preprocessor

    // Are there better alternatives than assigning the current timestamp as the exampleId?
    val exampleId: String = (System.currentTimeMillis/1000).toString
    val sempreExample = WikiTablesDataProcessor.makeCustomExample(options.valueOf(questionOpt),
      options.valueOf(tableStringOpt), exampleId)
    val pnpExample = WikiTablesUtil.convertCustomExampleToWikiTablesExample(sempreExample)
    val entityLinking = new WikiTablesEntityLinker().getEntityLinking(pnpExample)
    val contextValue = pnpExample.getContext()
    val graph = contextValue.graph.asInstanceOf[TableKnowledgeGraph]
    // Reusing example id  as the table id. The original idea of assigning table ids was to use them
    // to serialize them as json files. We don't need to do that at test time anyway.
    val table = Table.knowledgeGraphToTable(exampleId, graph)
    val processedExample = RawExample(pnpExample, entityLinking, table)
    WikiTablesUtil.preprocessExample(processedExample, parser.vocab, featureGenerator, typeDeclaration)
    val (testResult, denotations) = TestWikiTablesCli.test(Seq(processedExample.ex), parser,
        options.valueOf(beamSizeOpt), options.has(evaluateDpdOpt), false, typeDeclaration, comparator,
        lfPreprocessor, println, options.valueOf(scoresOutputOpt))
    val answers = if (options.has(numAnswersOpt)) {
      denotations.map { x => x._1 -> x._2.take(options.valueOf(numAnswersOpt)) }
    } else {
      denotations
    }
    println(answers)
    answers
  }

  def loadSerializedParser(modelFilename: String): SemanticParser = {
    val loader = new ModelLoader(modelFilename)
    val model = PnpModel.load(loader)
    val parser = SemanticParser.load(loader, model)
    loader.done()
    parser
  }
}

class ParserPredictionFunction

object TestWikiTablesCli {

  def main(args: Array[String]): Unit = {
    (new TestWikiTablesCli()).run(args)
  }
  
  /** Evaluate the test accuracy of parser on examples. Logical
   * forms are compared for equality using comparator.
   */
  def test(examples: Seq[WikiTablesExample], parser: SemanticParser, beamSize: Int,
      evaluateDpd: Boolean, evaluateOracle: Boolean, typeDeclaration: TypeDeclaration,
      comparator: ExpressionComparator, preprocessor: LfPreprocessor,
      print: Any => Unit, scoresOutputFp: String): (SemanticParserLoss, Map[String, List[(Value, Double)]]) = {

    print("")
    var numCorrect = 0
    var numCorrectAt10 = 0
    val exampleDenotations = MutableMap[String, List[(Value, Double)]]()
    
    val sw = new PrintWriter(new File(scoresOutputFp))

    for (e <- examples) {
      val sent = e.sentence
      print("example id: " + e.id +  " " + e.tableString)
      print(sent.getWords.asScala.mkString(" "))
      print(sent.getAnnotation("unkedTokens").asInstanceOf[List[String]].mkString(" "))

      val entityLinking = sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking]
      val dist = parser.parse(sent.getAnnotation("tokenIds").asInstanceOf[Array[Int]],
          entityLinking)

      ComputationGraph.renew()
      val context = PnpInferenceContext.init(parser.model)
      val results = dist.beamSearch(beamSize, 75, Env.init, context)

      // ksk added
      val states = results.executions.map(_.value.asInstanceOf[SemanticParserState])
//      val tokenEntityScores = states(0).getScoreMatrix() // The scores remain same for all states
//      printTokenEntityScores(entityLinking, e.sentence.getWords.asScala.toArray, tokenEntityScores, print)
      sw.write(e.id + "\t" + "www.cse.iitd.ac.in/~kskeshav/wikitables/"+e.id+"\n")
      sw.write("Question: "+sent.getWords.asScala.mkString(" ")+"\n")
      val tokenEntityScores = if(states.length > 0){
        states(0).getScoreMatrix() // The scores remain same for all states
      } else {
        None
      }

      if(tokenEntityScores != None){
        printTokenEntityScores(entityLinking, e.sentence.getWords.asScala.toArray, tokenEntityScores.asInstanceOf[Expression], sw, sent.getAnnotation("NER").asInstanceOf[List[List[String]]])
      }

      val beam = results.executions.slice(0, 10)
      val correctAndValue = beam.map { x =>
        val expression = x.value.decodeExpression
        val value = e.executeFormula(preprocessor.postprocess(expression))
        
        val isCorrect = if (evaluateDpd) {
          // Evaluate the logical forms using the output of dynamic programming on denotations.
          e.logicalForms.size > 0 && e.logicalForms.map(x => comparator.equals(x, expression)).reduce(_ || _)
        } else {
          // Evaluate the logical form by executing it.
          if (value.isDefined) {
            e.isValueCorrect(value.get)
          } else {
            false
          }
        }

        if (isCorrect) {
          print("* " + x.logProb.formatted("%02.3f") + "  " + expression + " -> " + value)
//          print("* " + x.logProb.formatted("%02.3f") + "  " + expression + " -> " + getAnswerTsvParts(value.toList).mkString("\t"))
          true
        } else {
          print("  " + x.logProb.formatted("%02.3f") + "  " + expression + " -> " + value)
//          print("  " + x.logProb.formatted("%02.3f") + "  " + expression + " -> " + getAnswerTsvParts(value.toList).mkString("\t"))
          false
        }
        
        (isCorrect, value, x.logProb)
      }
      
      var exampleCorrect = false  
      if (correctAndValue.length > 0) {
        if (correctAndValue(0)._1) {
          numCorrect += 1
          exampleCorrect = true
        }
      }
      if (correctAndValue.foldRight(false)((x, y) => x._1 || y)) {
        numCorrectAt10 += 1
      }

      // Store all defined values sorted in probability order
      exampleDenotations(e.id) = correctAndValue.filter(_._2.isDefined).map(
          x => (x._2.get, x._3)).toList
      
      print("id: " + e.id + " " + exampleCorrect)

      // Re-parse with a label oracle to find the highest-scoring correct parses.
      if (evaluateOracle) {
        val oracle = parser.getMultiLabelScore(e.logicalForms, entityLinking, typeDeclaration)
        if (oracle.isDefined) { 
          val oracleContext = PnpInferenceContext.init(parser.model).addExecutionScore(oracle.get)
          val oracleResults = dist.beamSearch(beamSize, 75, Env.init, oracleContext)
            
          oracleResults.executions.map { x =>
            val expression = x.value.decodeExpression
          
            if(e.goldLogicalForm.isDefined && expression == e.goldLogicalForm.get) {
              print("o " + x.logProb.formatted("%02.3f") + " [GOLD] " + expression)
            } else {
              print("o " + x.logProb.formatted("%02.3f") + "  " + expression)
            }
          }
        } else {
          print("  No correct logical forms in oracle.")
        }
      }

      // Print the attentions of the best predicted derivation
      if (beam.nonEmpty) {
        printAttentions(beam(0).value, e.sentence.getWords.asScala.toArray, print)
      }

      printEntityTokenFeatures(entityLinking, e.sentence.getWords.asScala.toArray, print)
    }

    sw.close()
    val loss = SemanticParserLoss(numCorrect, numCorrectAt10, examples.length)
    (loss, exampleDenotations.toMap)
  }

  def printAttentions(state: SemanticParserState, tokens: Array[String],
      print: Any => Unit): Unit = {
    val templates = state.getTemplates
    val attentions = state.getAttentions
    for (i <- 0 until templates.length) {
      val values = ComputationGraph.incrementalForward(attentions(i)).toSeq()
      val maxIndex = values.zipWithIndex.max._2

      val tokenStrings = for {
        j <- 0 until values.length
      } yield {
        val color = if (j == maxIndex) {
          Console.RED
        } else if (values(j) > 0.1) {
          Console.YELLOW
        } else {
          Console.RESET
        }
        color + tokens(j) + Console.RESET
      }

      print("  " + tokenStrings.mkString(" ") + " " + templates(i))
    }
  }

  def printTokenEntityScores(entityLinking: EntityLinking, tokens: Array[String],
                             tokenEntityScores: Expression, writer: PrintWriter, nerList: List[List[String]]): Unit = {
    writer.write("\n")
    val tokenEntityProbScores = Expression.transpose(Expression.softmax(Expression.transpose(tokenEntityScores)))

    print(nerList)
    for ((token, i) <- tokens.zipWithIndex) {
      writer.write(token+"\t")
      if(nerList(i) == Nil){
        writer.write("O"+"\t")
      }
      else{
        writer.write(nerList(i)(0)+"\t")
      }
      val scoresToken = ComputationGraph.forward(Expression.pick(tokenEntityProbScores, i, 0)).toVector()
      val (sortedScores, indices) = scoresToken.zipWithIndex.sorted.unzip
      var stop = 0
      for ((score, index) <- sortedScores.zip(indices).reverse) {
        stop = stop + 1
        if(stop <= 2){
          writer.write(entityLinking.entities(index).expr + ":" + score + "\t")
        }
      }
      writer.write("\n")
    }
    writer.write("\n")
  }

  def printEntityTokenFeatures(entityLinking: EntityLinking, tokens: Array[String],
      print: Any => Unit): Unit = {
    for ((entity, features) <- entityLinking.entities.zip(entityLinking.entityTokenFeatures)) {
      val dim = features._1
      val featureMatrix = features._2
      val values = Expression.input(dim, featureMatrix)
      
      for ((token, i) <- tokens.zipWithIndex) {
        val features = ComputationGraph.incrementalForward(Expression.pick(values, i)).toSeq
        if (features.filter(_ != 0.0f).size > 0) {
          print(entity.expr + " " + token + " " + features)
        }
      }
    }
  }
  
  def getAnswerTsvParts(values: List[Value]): List[String] = {
    // Don't return the empty string, because it's always wrong.
    // Empty string can come from null values or
    val valueStringLists = values.map(v => valueToStrings(v).map(tsvEscape).filter(_.length > 0))
    val nonEmptyLists = valueStringLists.filter(_.size > 0)
    nonEmptyLists.headOption.getOrElse(List())
  }

  def valueToStrings(value: Value): List[String] = {
    if (value.isInstanceOf[ListValue]) {
      val values = for {
        elt <- value.asInstanceOf[ListValue].values.asScala
        eltValue <- valueToString(elt)
        if eltValue != null
      } yield {
        eltValue
      }
      values.toList
    } else {
      valueToString(value).toList
    }
  }
  
  def valueToString(value: Value): Option[String] = {
    if (value.isInstanceOf[NameValue]) {
      Some(value.asInstanceOf[NameValue].description)
    } else if (value.isInstanceOf[NumberValue]) {
      Some(value.asInstanceOf[NumberValue].value.toString)
    } else if (value.isInstanceOf[DateValue]) {
      val d = value.asInstanceOf[DateValue]
      
      Some(datePartToString(d.year) + "-" + datePartToString(d.month) + "-" + datePartToString(d.day))
    } else {
      None
    }
  }
  
  def datePartToString(d: Int): String = {
    if (d == -1) {
      "xx"
    } else {
      d.toString
    }
  }
  
  // This makes strings printable in tsv format while also retaining correctness
  // of evaluate.py.
  def tsvEscape(s: String): String = {
    val newlineCount = s.count(c => c == '\n')
    val newlineFixed = if (newlineCount > 2) {
      s.split('\n')(0)
    } else {
      s.replaceAllLiterally("\n", " ")
    }
    newlineFixed.trim()
  }
}
