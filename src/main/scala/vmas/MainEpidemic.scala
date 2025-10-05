package vmas

import scarlib.model.*
import scarlib.neuralnetwork.DQNAbstractFactory
import vmas.RewardFunctionEpidemic.{CurrentState, InfectionPenalty, Lambda, NewState, RewardFunctionStep, Tensor, VaccinationDrive, airportFunc, hospitalUtilization, rewardFunctionStep}
import me.shadaj.scalapy.interpreter.CPythonInterpreter
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.PyQuote
import vmas.VMASEpidemicState.encoding
import vmas.WANDBLogger
import vmas.VmasEpidemicEnvironment
import scarlib.model.DSL.{CTDELearningSystem, actionSpace, agents, dataset, environment, learningConfiguration, rewardFunction}

import scala.concurrent.ExecutionContext
import scala.language.implicitConversions
import me.shadaj.scalapy.*
import me.shadaj.scalapy.py.SeqConverters
import ai.kien.python.Python
import org.apache.spark.sql.types._
import org.apache.spark.sql.{SparkSession, Row}
import org.apache.spark.sql.functions.*
import org.apache.spark.sql.Encoders
import scarlib.model.*
import org.json4s._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.write
import java.io.{File, FileWriter, BufferedWriter}

object MainEpidemic extends App {

  val spark = SparkSession.builder()
    .appName("EpidemicSimulation")
    .master("local[*]")
    .config("spark.driver.host", "localhost")
    .config("spark.driver.bindAddress", "0.0.0.0")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")
  import spark.implicits._

  val epidemicSchema = StructType(Seq(
    StructField("susceptible", IntegerType, nullable = false),
    StructField("infected", IntegerType, nullable = false),
    StructField("recovered", IntegerType, nullable = false),
    StructField("deaths", IntegerType, nullable = false),
    StructField("exposed", IntegerType, nullable = false),
    StructField("hospitalCapacity", IntegerType, nullable = false),
    StructField("location", StringType, nullable = false),
    StructField("airports", ArrayType(StringType), nullable = true),
    StructField("vaccinatedPopulation", IntegerType, nullable = false),
    StructField("travelVolume", IntegerType, nullable = false),
    StructField("currentDate", StringType, nullable = true),
    StructField("previousInfected", IntegerType, nullable = false),
    StructField("previousRecovered", IntegerType, nullable = false),
    StructField("previousDeaths", IntegerType, nullable = false),
    StructField("ageDistribution", MapType(StringType, IntegerType), nullable = true),
    StructField("incomingTravelers", MapType(StringType, IntegerType), nullable = true),
    StructField("outgoingTravelers", MapType(StringType, IntegerType), nullable = true),
    StructField("airportTraffic", MapType(StringType, MapType(StringType, IntegerType)), nullable = true)
  ))

  def epidemicDataToRow(data: EpidemicData): Row = {
    Row(
      data.susceptible, data.infected, data.recovered, data.deaths, data.exposed,
      data.hospitalCapacity, data.location, data.airports, data.vaccinatedPopulation,
      data.travelVolume, data.currentDate, data.previousInfected, data.previousRecovered,
      data.previousDeaths, data.ageDistribution, data.incomingTravelers,
      data.outgoingTravelers, data.airportTraffic
    )
  }

  val epidemicData = Seq(
    EpidemicData(
      susceptible = 1000000, infected = 1500, recovered = 500, deaths = 50, exposed = 200,
      hospitalCapacity = 10000, location = "Italy", airports = List("FCO", "MXP"),
      vaccinatedPopulation = 50000, travelVolume = 25000
    ),
    EpidemicData(
      susceptible = 999000, infected = 1200, recovered = 400, deaths = 30, exposed = 150,
      hospitalCapacity = 8000, location = "Germany", airports = List("FRA", "MUC"),
      vaccinatedPopulation = 45000, travelVolume = 30000
    ),
    EpidemicData(
      susceptible = 11100000, infected = 3000, recovered = 50, deaths = 1000, exposed = 400,
      hospitalCapacity = 100000, location = "China", airports = List("XIV", "JKP", "MNO", "TUV"),
      vaccinatedPopulation = 5000, travelVolume = 2500
    )
  )

  val rowData = epidemicData.map(epidemicDataToRow)
  val rdd = spark.sparkContext.parallelize(rowData)
  val epidemicDF = spark.createDataFrame(rdd, epidemicSchema)
  val collectedData = epidemicDF.collect()

  implicit val formats: Formats = DefaultFormats
  val jsonData = collectedData.map { row => (0 until row.length).map(row.get).toList }
  val jsonString = write(jsonData)

  val nAgents = 3
  val nSteps = 50
  val nEpochs = 100

  // Initialize Python
  CPythonInterpreter.execManyLines("import torch")
  CPythonInterpreter.execManyLines("import numpy as np")

  val diseaseOrigin = "China"
  val targetCountries = Seq("Italy", "Germany")

  val epidemicRewardFunction =
    InfectionPenalty((Tensor(0.5)), CurrentState) ++
      hospitalUtilization(Tensor(-0.2), CurrentState) ++
      VaccinationDrive(Tensor(-0.8), NewState) ++
      airportFunc(Tensor(0.5), diseaseOrigin, targetCountries, CurrentState) -->
      Lambda("x: x.sum()") >>
      Lambda("x: x.clamp(-100.0, 100.0)")

  println(s"Epidemic Reward Function DSL: ${epidemicRewardFunction.toString}")

  rewardFunctionStep { epidemicRewardFunction }

  val descriptor = VmasStateDescriptor(hasPosition = false, hasVelocity = false, extraDimension = 7)
  VMASEpidemicState.setDescriptor(descriptor)
  println(s"Epidemic state encoding size: ${VMASEpidemicState.encoding.elements()}")

  // Python reward function - CRITICAL: No indentation after opening """
  val rewardFunctionCode = """def epidemic_rf(env, agent):
    import torch
    agent_id = int(agent.name.split("_")[1])
    if not hasattr(env, 'epidemic_states') or agent_id >= len(env.epidemic_states):
        return torch.zeros(env.world.batch_dim, dtype=torch.float32, device=env.world.device)
    state = env.epidemic_states[agent_id]
    batch_size = env.world.batch_dim if hasattr(env.world, 'batch_dim') else 1
    infection_rate = state.infected / max(1.0, (state.susceptible + state.infected + state.recovered))
    infection_penalty = -0.5 * infection_rate * 100
    hospital_util = state.infected / max(1.0, state.hospital_capacity)
    if hospital_util > 1.0:
        hospital_penalty = -0.3 * hospital_util * 100
    else:
        hospital_penalty = -0.3 * hospital_util * 2
    vaccination_rate = state.vaccinated_population / max(1.0, (state.susceptible + state.infected + state.recovered))
    vaccination_reward = 0.8 * vaccination_rate * 100
    airport_penalty = -0.2 * (state.airports / 10.0) * infection_rate
    total_reward = infection_penalty + hospital_penalty + vaccination_reward + airport_penalty
    total_reward = max(-100.0, min(100.0, total_reward))
    return torch.full((batch_size,), total_reward, device=env.world.device, dtype=torch.float32)
"""
  CPythonInterpreter.execManyLines(rewardFunctionCode)
  val rfLambda = py.Dynamic.global.epidemic_rf

  // Observation function with DYNAMIC epidemic simulation
  val obsFromSparkCode = s"""import json
import torch

row_data = json.loads('$jsonString')

def epidemic_obs_from_spark(env, agent):
    agent_id = int(agent.name.split("_")[1])

    # Initialize epidemic state on first call
    if not hasattr(env, 'epidemic_state'):
        env.epidemic_state = {}
        for idx, obs_values in enumerate(row_data):
            env.epidemic_state[idx] = {
                'susceptible': float(obs_values[0]),
                'infected': float(obs_values[1]),
                'recovered': float(obs_values[2]),
                'deaths': float(obs_values[3]),
                'hospital_capacity': float(obs_values[5]),
                'vaccinated': float(obs_values[8]),
                'airports': len(obs_values[7]),
                'last_action': None
            }

    if agent_id >= len(row_data):
        default_obs = torch.zeros(7, dtype=torch.float32, device=env.world.device)
        agent.obs = default_obs
        return default_obs.unsqueeze(0).repeat(env.world.batch_dim, 1)

    state = env.epidemic_state[agent_id]

    # Simulate epidemic dynamics based on last action
    if hasattr(agent, 'last_action') and agent.last_action is not None:
        action_idx = agent.last_action

        # Base transmission rate
        beta = 0.3
        gamma = 0.1

        # Action effects on transmission
        if action_idx == 1:
            beta *= 0.7
        elif action_idx == 3:
            beta *= 0.5
        elif action_idx == 8:
            state['vaccinated'] += state['susceptible'] * 0.05
            state['susceptible'] -= state['susceptible'] * 0.05
        elif action_idx == 7:
            state['vaccinated'] += state['susceptible'] * 0.02
            state['susceptible'] -= state['susceptible'] * 0.02
        elif action_idx == 5:
            gamma *= 1.5

        # SIR model dynamics
        total_pop = state['susceptible'] + state['infected'] + state['recovered']
        new_infections = beta * state['susceptible'] * state['infected'] / max(1, total_pop)
        new_recoveries = gamma * state['infected']

        # Death rate increases if hospitals overwhelmed
        death_rate = 0.01
        if state['infected'] > state['hospital_capacity']:
            death_rate = 0.05
        new_deaths = death_rate * state['infected']

        # Update state
        state['susceptible'] = max(0, state['susceptible'] - new_infections)
        state['infected'] = max(0, state['infected'] + new_infections - new_recoveries - new_deaths)
        state['recovered'] += new_recoveries
        state['deaths'] += new_deaths

    # Create observation tensor
    obs = torch.tensor([
        state['susceptible'] / 1000000.0,
        state['infected'] / 10000.0,
        state['recovered'] / 10000.0,
        state['deaths'] / 1000.0,
        state['hospital_capacity'] / 20000.0,
        state['vaccinated'] / 1000000.0,
        state['airports'] / 10.0
    ], dtype=torch.float32, device=env.world.device)

    # For vectorized environments, create batch dimension
    if env.world.batch_dim > 1:
        obs = obs.unsqueeze(0).expand(env.world.batch_dim, -1)
    else:
        obs = obs.unsqueeze(0)

    agent.obs = obs
    return obs
"""
  CPythonInterpreter.execManyLines(obsFromSparkCode)
  val obsLambda = py.Dynamic.global.epidemic_obs_from_spark

  // Initialize logging
  WANDBLogger.init()

  // Add Python path
  CPythonInterpreter.execManyLines(
    """import sys, os
sys.path.extend([os.path.abspath(p) for p in ["./src/main/resources","./build/resources/main","./src/main/scala/resources"] if os.path.isdir(p) and os.path.abspath(p) not in sys.path])
"""
  )

  val scenario = py.module("AbstractEnv").Scenario(rfLambda, obsLambda)

  val envSettings = VmasSettings(
    scenario = scenario,
    nEnv = 1,  // Start with 1 env - simpler for debugging
    nAgents = nAgents,
    nTargets = 0,
    nSteps = nSteps,
    nEpochs = nEpochs,
    device = "cpu"
  )

  implicit val configuration: Environment => Unit = (e: Environment) => {
    val env = e.asInstanceOf[VmasEpidemicEnvironment]
    env.setSettings(envSettings)
    env.setLogger(WANDBLogger)
    env.enableRender(false)
    env.initEnv()
  }

  val where = "./epidemic_networks"

  val epidemicSystem = CTDELearningSystem {
    rewardFunction { DSLRewardFunction() }
    actionSpace { RealEpidemicAction.toSeq }
    dataset { ReplayBuffer[State, Action](10000) }
    agents { nAgents }
    learningConfiguration {
      LearningConfiguration(
        dqnFactory = new EpidemicNNFactory(VMASEpidemicState.encoding.elements(), RealEpidemicAction.toSeq),
        snapshotPath = where
      )
    }
    environment { "vmas.VmasEpidemicEnvironment" }
  }(ExecutionContext.global, VMASEpidemicState.encoding)

  println("Starting epidemic simulation training...")
  for (epoch <- 1 to nEpochs) {
    if (epoch % 10 == 0) {
      println(s"Epoch $epoch/$nEpochs")
    }
    epidemicSystem.learn(1, nSteps)
  }
  println("Training completed.")

  // Verify Python and PyTorch
  CPythonInterpreter.execManyLines("""print(">>> Python alive")""")
  val torch = py.module("torch")
  val t = torch.tensor(Seq(1, 2, 3).toPythonCopy)
  println(s">>> Torch tensor from Scala: $t")

  println("\n" + "="*80)
  println("STARTING MODEL VALIDATION")
  println("="*80)

  try {
    // Store validation results in Python global variables
    CPythonInterpreter.execManyLines("""
import glob
import torch
import torch.nn as nn
import os

validation_results = {}

files = glob.glob("epidemic_networks/*")
validation_results['num_files'] = len(files)
validation_results['files'] = files

if files:
    latest = max(files, key=os.path.getctime)
    validation_results['latest_checkpoint'] = latest

    checkpoint = torch.load(latest, map_location="cpu")
    model = nn.Sequential(
        nn.Linear(7, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 9)
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    action_names = ['NoAction', 'SocialDistancing', 'NoTravelRestriction',
                   'CompleteTravelLockdown', 'NormalHealthcare',
                   'EmergencyHealthcareMobilization', 'NoVaccination',
                   'TargetedVaccination', 'MassVaccination']

    # Test scenarios
    scenarios = {
        'Italy (high infection)': [1.0, 0.15, 0.05, 0.005, 0.5, 0.05, 0.2],
        'Germany (moderate)': [0.999, 0.12, 0.04, 0.003, 0.4, 0.045, 0.2],
        'China (severe)': [11.1, 0.3, 0.005, 0.1, 5.0, 0.005, 0.4],
        'Controlled (low infection)': [0.3, 0.001, 0.65, 0.001, 0.5, 0.8, 0.2]
    }

    validation_results['predictions'] = {}
    for name, obs in scenarios.items():
        obs_tensor = torch.tensor([obs])
        output = model(obs_tensor).detach().numpy()[0]
        best_idx = int(output.argmax())
        validation_results['predictions'][name] = {
            'action': action_names[best_idx],
            'action_idx': best_idx,
            'q_values': output.tolist()
        }

    validation_results['success'] = True
else:
    validation_results['success'] = False
""")

    // Retrieve results from Python
    val validationResults = py.Dynamic.global.validation_results
    val numFiles = validationResults.bracketAccess("num_files").as[Int]
    val success = validationResults.bracketAccess("success").as[Boolean]

    println(s"Found $numFiles checkpoint files")

    if (success) {
      val latestCheckpoint = validationResults.bracketAccess("latest_checkpoint").as[String]
      println(s"Loaded checkpoint: $latestCheckpoint")
      println("\nModel Predictions:")
      println("-" * 80)

      val predictions = validationResults.bracketAccess("predictions")
      val scenarios = Seq("Italy (high infection)", "Germany (moderate)", "China (severe)", "Controlled (low infection)")

      scenarios.foreach { scenario =>
        val pred = predictions.bracketAccess(scenario)
        val action = pred.bracketAccess("action").as[String]
        val actionIdx = pred.bracketAccess("action_idx").as[Int]
        val qValues = pred.bracketAccess("q_values")

        println(s"\n$scenario:")
        println(s"  Recommended Action: $action (index: $actionIdx)")

        // Get top 3 Q-values
        val qList = (0 until 9).map { i =>
          val qVal = qValues.bracketAccess(i).as[Double]
          (i, qVal)
        }.sortBy(-_._2).take(3)

        val actionNames = Seq("NoAction", "SocialDistancing", "NoTravelRestriction",
          "CompleteTravelLockdown", "NormalHealthcare",
          "EmergencyHealthcareMobilization", "NoVaccination",
          "TargetedVaccination", "MassVaccination")

        println("  Top 3 Actions:")
        qList.foreach { case (idx, qVal) =>
          println(f"    ${actionNames(idx)}: $qVal%.3f")
        }
      }

      println("\n" + "="*80)
      println("VALIDATION ANALYSIS")
      println("="*80)

      // Check if model makes sense
      val italyAction = predictions.bracketAccess("Italy (high infection)").bracketAccess("action").as[String]
      val chinaAction = predictions.bracketAccess("China (severe)").bracketAccess("action").as[String]
      val controlledAction = predictions.bracketAccess("Controlled (low infection)").bracketAccess("action").as[String]

      println(s"\nHigh infection scenario → $italyAction")
      println(s"Severe outbreak scenario → $chinaAction")
      println(s"Controlled scenario → $controlledAction")

      if (italyAction == chinaAction && chinaAction == controlledAction) {
        println("\n⚠ WARNING: Model recommends same action for all scenarios")
        println("   This suggests the model has NOT learned meaningful policies")
      } else {
        println("\n✓ Model produces different actions for different scenarios")
        println("   This suggests learning may have occurred")
      }

    } else {
      println("No checkpoints found - model did not save during training")
    }

    println("\n" + "="*80)
    println("Validation completed")
    println("="*80)

  } catch {
    case e: Exception =>
      println(s"ERROR during validation: ${e.getMessage}")
      e.printStackTrace()
  }

  println("Shutting down...")
  spark.stop()
  println("Shutdown complete")
}

case class DSLRewardFunction() extends RewardFunction {
  override def compute(currentState: State, action: Action, newState: State): Double = {
    val reward = RewardFunctionDSL.rf match {
      case Some(r) => r.compute(currentState, action, newState)
      case None => 0.0
    }
    if (scala.util.Random.nextDouble() < 0.01) {
      println(s"Reward: $reward for action: $action")
    }
    reward
  }
}

case class DebugRewardFunction() extends RewardFunction {
  override def compute(currentState: State, action: Action, newState: State): Double = -math.random()
}