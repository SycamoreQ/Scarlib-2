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
    .getOrCreate()

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
      data.susceptible,
      data.infected,
      data.recovered,
      data.deaths,
      data.exposed,
      data.hospitalCapacity,
      data.location,
      data.airports, // List[String] -> Seq[String] for Spark
      data.vaccinatedPopulation,
      data.travelVolume,
      data.currentDate,
      data.previousInfected,
      data.previousRecovered,
      data.previousDeaths,
      data.ageDistribution, // Map[String, Int]
      data.incomingTravelers, // Map[String, Int]
      data.outgoingTravelers, // Map[String, Int]
      data.airportTraffic // Map[String, Map[String, Int]]
    )
  }


  val epidemicData = Seq(
    EpidemicData(
      susceptible = 1000000,
      infected = 1500,
      recovered = 500,
      deaths = 50,
      exposed = 200,
      hospitalCapacity = 10000,
      location = "Italy",
      airports = List("FCO", "MXP"),
      vaccinatedPopulation = 50000,
      travelVolume = 25000
    ),
    EpidemicData(
      susceptible = 999000,
      infected = 1200,
      recovered = 400,
      deaths = 30,
      exposed = 150,
      hospitalCapacity = 8000,
      location = "Germany",
      airports = List("FRA", "MUC"),
      vaccinatedPopulation = 45000,
      travelVolume = 30000
    ),
    EpidemicData(
      susceptible = 11100000,
      infected = 3000,
      recovered = 50,
      deaths = 1000,
      exposed = 400,
      hospitalCapacity = 100000,
      location = "China",
      airports = List("XIV", "JKP", "MNO", "TUV"),
      vaccinatedPopulation = 5000,
      travelVolume = 2500,
    )
  )

  val rowData = epidemicData.map(epidemicDataToRow)
  val rdd = spark.sparkContext.parallelize(rowData)
  val epidemicDF = spark.createDataFrame(rdd, epidemicSchema)

  val collectedData = epidemicDF.collect()

  // Make sure this implicit is in scope
  implicit val formats: Formats = DefaultFormats

  val jsonData = collectedData.map { row =>
    (0 until row.length).map(row.get).toList
  }

  val jsonString = write(jsonData) // valid JSON string


  val nAgents = 3  // Number of countries/regions in epidemic simulation
  val nSteps = 100
  val nEpochs = 150

  // Initialize Python environment
  CPythonInterpreter.execManyLines("import torch")
  CPythonInterpreter.execManyLines("import numpy as np")

  // Define epidemic-specific constants
  val diseaseOrigin = "China"
  val targetCountries = Seq("Italy" ,  "Germany")

  val epidemicRewardFunction =
    InfectionPenalty((Tensor(0.5)), CurrentState) ++
      hospitalUtilization(Tensor(-0.2), CurrentState) ++
      VaccinationDrive(Tensor(-0.8), NewState) ++
      airportFunc(Tensor(0.5) , diseaseOrigin, targetCountries, CurrentState) -->
      Lambda("x: x.sum()") >>
      Lambda("x: x.clamp(-100.0, 100.0)")

  println(s"Epidemic Reward Function DSL: ${epidemicRewardFunction.toString}")


  // Set up reward function step using your DSL (similar to Main.scala)
  rewardFunctionStep {
    epidemicRewardFunction
  }

  val descriptor = VmasStateDescriptor(
    hasPosition = false,
    hasVelocity = false,
    extraDimension = 7
  )
  VMASEpidemicState.setDescriptor(descriptor)
  println(s"Epidemic state encoding size: ${VMASEpidemicState.encoding.elements()}")


  // Define epidemic reward function in Python using DSL components
  CPythonInterpreter.execManyLines(
    """def epidemic_rf(env, agent):
            import torch
            import math

            # Get agent state
            agent_id = int(agent.name.split("_")[1])

            if not hasattr(env, 'epidemic_states') or agent_id >= len(env.epidemic_states):
                return torch.zeros(env.world.batch_dim, dtype=torch.float32, device=env.world.device)

            state = env.epidemic_states[agent_id]

            # Implementation of your DSL functions:
            # InfectionPenalty(-0.5, CurrentState)
            infection_rate = getattr(state, 'infection_rate', state.infected / (state.susceptible + state.infected + state.recovered))
            infection_penalty = -0.5 * infection_rate * 100

            # hospitalUtilization(-0.3, CurrentState)
            hospital_util = state.infected / state.hospital_capacity if state.hospital_capacity > 0 else 0
            if hospital_util > 10000:
                hospital_penalty = -0.3 * hospital_util * 100
            elif hospital_util > 1000:
                hospital_penalty = -0.3 * hospital_util * 10
            else:
                hospital_penalty = -0.3 * hospital_util * 2

            # VaccinationDrive(0.8, NewState) - using current state as proxy
            vaccination_rate = getattr(state, 'vaccination_rate', state.vaccinated_population / (state.susceptible + state.infected + state.recovered))
            if vaccination_rate > infection_rate:
                vaccination_reward = 0.8 * vaccination_rate * 100
            else:
                vaccination_reward = 0.8 * vaccination_rate * 2

            # airportFunc(-0.2, diseaseOrigin, targetCountries, CurrentState)
            # Simplified airport connectivity penalty
            airport_penalty = -0.2 * (state.airports / 10.0) * infection_rate

            # Combine all components (DSL: ++ operations)
            total_reward = infection_penalty + hospital_penalty + vaccination_reward + airport_penalty

            # Apply DSL transformations: --> Lambda("x: x.sum()") >> Lambda("x: x.clamp(-100.0, 100.0)")
            total_reward = max(-100.0, min(100.0, total_reward))  # Clamp between -100 and 100

            return torch.tensor(total_reward, device=env.world.device, dtype=torch.float32).squeeze(0)
        """)

  val rfLambda = py.Dynamic.global.epidemic_rf

  // uses random data to initialize epidemic data
  CPythonInterpreter.execManyLines(
    """def epidemic_obs(env, agent):
            import torch

            # Initialize epidemic states if needed (similar to distance calculation in Main.scala)
            if agent.name == "agent_0":
                # Initialize epidemic data for all agents
                env.epidemic_states = []
                for i in range(env.n_agents):
                    epidemic_state = {
                        'susceptible': 1000000.0 - (i * 1000),  # Example population
                        'infected': 100.0 + (i * 10),
                        'recovered': 50.0 + (i * 5),
                        'deaths': 5.0 + i,
                        'hospital_capacity': 10000.0,
                        'vaccinated_population': 5000.0 + (i * 100),
                        'airports': 5 + i
                    }
                    env.epidemic_states.append(type('EpidemicState', (), epidemic_state)())

            # Get agent-specific observation
            agent_id = int(agent.name.split("_")[1])

            if hasattr(env, 'epidemic_states') and agent_id < len(env.epidemic_states):
                state = env.epidemic_states[agent_id]

                # Create observation tensor (normalized values)
                obs = torch.tensor([
                    state.susceptible / 1000000.0,      # Normalized susceptible population
                    state.infected / 10000.0,           # Normalized infected population
                    state.recovered / 10000.0,          # Normalized recovered population
                    state.deaths / 1000.0,              # Normalized deaths
                    state.hospital_capacity / 20000.0,  # Normalized hospital capacity
                    state.vaccinated_population / 1000000.0, # Normalized vaccinated
                    state.airports / 10.0               # Normalized airports
                ], dtype=torch.float32, device=env.world.device)

                agent.obs = obs
                return obs.unsqueeze(0)

            # Default observation if no epidemic state
            default_obs = torch.zeros(7, dtype=torch.float32, device=env.world.device)
            agent.obs = default_obs
            return default_obs.unsqueeze(0)
        """)


  ///Uses spark to initialize epidemic data
  CPythonInterpreter.execManyLines(
    s"""
       |import json, torch
       |
       |row_data = json.loads('''$jsonString''')
       |
       |def epidemic_obs_from_spark(env, agent):
       |    agent_id = int(agent.name.split("_")[1])
       |
       |    if agent_id < len(row_data):
       |        obs_values = row_data[agent_id]
       |
       |        obs = torch.tensor([
       |            obs_values[0] / 1000000.0,
       |            obs_values[1] / 10000.0,
       |            obs_values[2] / 10000.0,
       |            obs_values[3] / 1000.0,
       |            obs_values[5] / 20000.0,
       |            obs_values[8] / 1000000.0,
       |            len(obs_values[7]) / 10.0  # airports count
       |        ], dtype=torch.float32, device=env.world.device)
       |
       |        agent.obs = obs
       |        return obs.unsqueeze(0)
       |
       |    default_obs = torch.zeros(7, dtype=torch.float32, device=env.world.device)
       |    agent.obs = default_obs
       |    return default_obs.unsqueeze(0)
       |""".stripMargin
  )


  val obsLambda = py.Dynamic.global.epidemic_obs_from_spark

  // Initialize logging
  WANDBLogger.init()

  // Ensure Python can import local modules like AbstractEnv.py
  CPythonInterpreter.execManyLines(
    """import sys, os; [sys.path.append(os.path.abspath(p)) for p in ["./src/main/resources","./build/resources/main","./src/main/scala/resources"] if os.path.isdir(p) and os.path.abspath(p) not in sys.path]"""
  )

  val scenario = py.module("AbstractEnv").Scenario(rfLambda, obsLambda)


  private val envSettings = VmasSettings(
    scenario = scenario,
    nEnv = 1,
    nAgents = nAgents,
    nTargets = 0,
    nSteps = nSteps,
    nEpochs = nEpochs,
    device = "cpu",
  )

  // Environment configuration
  implicit val configuration: Environment => Unit = (e: Environment) => {
    val env = e.asInstanceOf[VmasEpidemicEnvironment]
    env.setSettings(envSettings)
    env.setLogger(WANDBLogger)
    env.enableRender(false)
    env.initEnv()
  }

  private val where = s"./epidemic_networks"

  val epidemicSystem = CTDELearningSystem {
    rewardFunction {
      EmptyRewardFunctionEpidemic()
    }

    actionSpace {
      RealEpidemicAction.toSeq
    }

    dataset {
      ReplayBuffer[State, Action](10000)
    }

    agents {
      nAgents
    }
    learningConfiguration {
      LearningConfiguration(
        dqnFactory = new EpidemicNNFactory(VMASEpidemicState.encoding.elements() , RealEpidemicAction.toSeq),
        snapshotPath = where
      )
    }

    environment {
      "vmas.VmasEpidemicEnvironment"
    }
  } (ExecutionContext.global, VMASEpidemicState.encoding)

  println("Starting epidemic simulation training...")
  epidemicSystem.learn(envSettings.nEpochs, envSettings.nSteps)

  CPythonInterpreter.execManyLines(
    s"""
       |import torch
       |import torch.nn as nn
       |import json
       |
       |# Same EpidemicModel as in EpidemicNNFactory (7 -> 64 -> 64 -> actions)
       |class EpidemicModel(nn.Module):
       |    def __init__(self, input_dim=7, hidden_dim=64, output_dim=${RealEpidemicAction.toSeq.size}):
       |        super().__init__()
       |        self.fc1 = nn.Linear(input_dim, hidden_dim)
       |        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
       |        self.fc3 = nn.Linear(hidden_dim, output_dim)
       |
       |    def forward(self, x):
       |        x = torch.relu(self.fc1(x))
       |        x = torch.relu(self.fc2(x))
       |        return self.fc3(x)
       |
       |# Load weights
       |model = EpidemicModel()
       |state_dict = torch.load("epidemic_networks/1-2025-09-23-19-27-03-agent-0", map_location="cpu")
       |model.load_state_dict(state_dict)
       |model.eval()
       |
       |# Load epidemic Spark data that Scala wrote into jsonString
       |row_data = json.loads('''$jsonString''')
       |
       |def normalize_row(obs_values):
       |    return [
       |        obs_values[0] / 1_000_000.0,   # susceptible
       |        obs_values[1] / 10_000.0,      # infected
       |        obs_values[2] / 10_000.0,      # recovered
       |        obs_values[3] / 1_000.0,       # deaths
       |        obs_values[5] / 20_000.0,      # hospitalCapacity
       |        obs_values[8] / 1_000_000.0,   # vaccinatedPopulation
       |        len(obs_values[7]) / 10.0      # airport count
       |    ]
       |
       |# Run inference for each country/agent
       |for idx, obs_values in enumerate(row_data):
       |    obs_tensor = torch.tensor([normalize_row(obs_values)], dtype=torch.float32)
       |    q_values = model(obs_tensor).detach().numpy().flatten().tolist()
       |    best_action = int(torch.argmax(model(obs_tensor)))
       |    print(f"Agent {idx}: Obs={normalize_row(obs_values)} -> Q-values={q_values}, BestAction={best_action}")
       |""".stripMargin
  )


}




case class EmptyRewardFunctionEpidemic() extends RewardFunction{
  override def compute(currentState: State, action: Action, newState: State): Double = 0.0
}

