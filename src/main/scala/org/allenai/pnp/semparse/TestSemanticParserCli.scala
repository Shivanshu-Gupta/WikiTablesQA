package org.allenai.pnp.semparse

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer
import org.allenai.pnp.{Env, PnpInferenceContext, PnpModel}

import com.jayantkrish.jklol.ccg.CcgExample
import com.jayantkrish.jklol.ccg.cli.TrainSemanticParser
import com.jayantkrish.jklol.ccg.lambda.TypeDeclaration
import com.jayantkrish.jklol.ccg.lambda2.ExpressionComparator
import com.jayantkrish.jklol.ccg.lambda2.ExpressionSimplifier
import com.jayantkrish.jklol.ccg.lambda2.SimplificationComparator
import com.jayantkrish.jklol.cli.AbstractCli
import com.jayantkrish.jklol.experiments.geoquery.GeoqueryUtil
import com.jayantkrish.jklol.training.NullLogFunction
import edu.cmu.dynet._
import joptsimple.OptionParser
import joptsimple.OptionSet
import joptsimple.OptionSpec


class TestSemanticParserCli extends AbstractCli() {
  
  var entityDataOpt: OptionSpec[String] = null
  var testDataOpt: OptionSpec[String] = null
  var modelOpt: OptionSpec[String] = null

  var beamSizeOpt: OptionSpec[Integer] = null
  
  override def initializeOptions(parser: OptionParser): Unit = {
    entityDataOpt = parser.accepts("entityData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    testDataOpt = parser.accepts("testData").withRequiredArg().ofType(classOf[String]).withValuesSeparatedBy(',').required()
    modelOpt = parser.accepts("model").withRequiredArg().ofType(classOf[String]).required()
    
    beamSizeOpt = parser.accepts("beamSize").withRequiredArg().ofType(classOf[Integer]).defaultsTo(5)
  }

  override def run(options: OptionSet): Unit = {
    Initialize.initialize(SemanticParserUtils.DYNET_PARAMS)
    
    // Initialize expression processing for Geoquery logical forms. 
    val typeDeclaration = GeoqueryUtil.getSimpleTypeDeclaration()
    val simplifier = GeoqueryUtil.getExpressionSimplifier
    val comparator = new SimplificationComparator(simplifier)
    
    val entityData = ListBuffer[CcgExample]()
    for (filename <- options.valuesOf(entityDataOpt).asScala) {
      entityData ++= TrainSemanticParser.readCcgExamples(filename).asScala
    }

    val testData = ListBuffer[CcgExample]()
    if (options.has(testDataOpt)) {
      for (filename <- options.valuesOf(testDataOpt).asScala) {
        testData ++= TrainSemanticParser.readCcgExamples(filename).asScala
      }
    }
    println(testData.size + " test examples")

    val loader = new ModelLoader(options.valueOf(modelOpt))
    val model = PnpModel.load(loader)
    val parser = SemanticParser.load(loader, model)
    loader.done()

    val vocab = parser.vocab

    val entityDict = TrainSemanticParserCli.buildEntityDictionary(entityData,
        vocab, typeDeclaration)
    
    val testPreprocessed = testData.map(x =>
      TrainSemanticParserCli.preprocessExample(x, simplifier, vocab, entityDict))

    println("*** Running Evaluation ***")
    val results = test(testPreprocessed, parser, options.valueOf(beamSizeOpt),
        typeDeclaration, simplifier, comparator, println)
  }
  
  /** Evaluate the test accuracy of parser on examples. Logical
   *  forms are compared for equality using comparator.  
   */
  def test(examples: Seq[CcgExample], parser: SemanticParser, beamSize: Int,
      typeDeclaration: TypeDeclaration, simplifier: ExpressionSimplifier,
      comparator: ExpressionComparator, print: String => Unit): SemanticParserLoss = {
    print("")
    var numCorrect = 0
    var numCorrectAt10 = 0
    for (e <- examples) {
      ComputationGraph.renew()
      val context = PnpInferenceContext.init(parser.model)

      print(e.getSentence.getWords.asScala.mkString(" "))
      print(e.getSentence.getAnnotation("originalTokens").asInstanceOf[List[String]].mkString(" "))
      print("expected: " + e.getLogicalForm)
      
      val sent = e.getSentence
      val dist = parser.parse(
          sent.getAnnotation("tokenIds").asInstanceOf[Array[Int]],
          sent.getAnnotation("entityLinking").asInstanceOf[EntityLinking])
      val results = dist.beamSearch(beamSize, 100, Env.init, context)
          
      val beam = results.executions.slice(0, 10)
      val correct = beam.map { x =>
        val simplified = simplifier.apply(x.value.decodeExpression)
        if (comparator.equals(e.getLogicalForm, simplified)) {
          print("* " + x.logProb.formatted("%02.3f") + "  " + simplified)
          true
        } else {
          print("  " + x.logProb.formatted("%02.3f") + "  " + simplified)
          false
        }
      }
      
      if (correct.length > 0 && correct(0)) {
        numCorrect += 1
      }
      if (correct.fold(false)(_ || _)) {
        numCorrectAt10 += 1
      }
      
      // Print the attentions of the best predicted derivation
      if (beam.length > 0) {
        val state = beam(0).value
        val templates = state.getTemplates
        val attentions = state.getAttentions
        val tokens = e.getSentence.getWords.asScala.toArray
        for (i <- 0 until templates.length) {
          val floatVector = ComputationGraph.getValue(attentions(i)).toVector
          val values = for {
            j <- 0 until floatVector.size
          } yield {
            floatVector(j)
          }
        
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
    }
    
    val loss = SemanticParserLoss(numCorrect, numCorrectAt10, examples.length)
    print(loss.toString)
    loss
  }
}

case class SemanticParserLoss(numCorrect: Int, oracleNumCorrect: Int, numExamples: Int) {
  val accuracy: Double = numCorrect.asInstanceOf[Double] / numExamples
  val oracleAccuracy: Double = oracleNumCorrect.asInstanceOf[Double] / numExamples
  
  override def toString(): String = {
    "accuracy: " + accuracy + " " + numCorrect + " / " + numExamples + "\n" +
    "oracle  : " + oracleAccuracy + " " + oracleNumCorrect + " / " + numExamples  
  }
}

object TestSemanticParserCli {
  def main(args: Array[String]): Unit = {
    (new TestSemanticParserCli()).run(args)
  }
}