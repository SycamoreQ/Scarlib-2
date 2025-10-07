package vmas

import scarlib.model.*
import me.shadaj.scalapy.interpreter.CPythonInterpreter
import me.shadaj.scalapy.py
import vmas.WANDBLogger
import vmas.VmasEpidemicEnvironment
import scarlib.model.DSL.{CTDELearningSystem, actionSpace, agents, dataset, environment, learningConfiguration, rewardFunction}
import scala.concurrent.ExecutionContext
import me.shadaj.scalapy.py.SeqConverters
import org.apache.spark.sql.SparkSession
import org.json4s._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.write

object MainEpidemic extends App {

  // ============================================================================
  // SPARK SETUP (for data loading only)
  // ============================================================================

  val spark = SparkSession.builder()
    .appName("EpidemicSimulation")
    .master("local[*]")
    .config("spark.driver.host", "localhost")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")

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

  implicit val formats: Formats = DefaultFormats
  val jsonString = write(epidemicData.map(d => Map(
    "susceptible" -> d.susceptible,
    "infected" -> d.infected,
    "recovered" -> d.recovered,
    "deaths" -> d.deaths,
    "hospital_capacity" -> d.hospitalCapacity,
    "location" -> d.location,
    "airports" -> d.airports.length,
    "vaccinated" -> d.vaccinatedPopulation
  )))

  // ============================================================================
  // CONFIGURATION
  // ============================================================================

  val nAgents = 3
  val nSteps = 100
  val nEpochs = 10

  println("="*80)
  println("EPIDEMIC SIMULATION CONFIGURATION")
  println("="*80)
  println(s"Agents: $nAgents (Italy, Germany, China)")
  println(s"Steps per epoch: $nSteps")
  println(s"Total epochs: $nEpochs")
  println(s"Total training steps: ${nAgents * nSteps * nEpochs}")
  println("="*80)

  CPythonInterpreter.execManyLines("import torch")
  CPythonInterpreter.execManyLines("import numpy as np")

  // ============================================================================
  // COMPLETE EPIDEMIC SIMULATION IN PYTHON
  // ============================================================================

  val epidemicSimulationCode = s"""
import json
import torch

# Load initial epidemic data
initial_data = json.loads('$jsonString')

# Action definitions
ACTION_EFFECTS = {
    0: {'name': 'NoAction', 'beta_mult': 1.0, 'gamma_mult': 1.0, 'vax_rate': 0.0},
    1: {'name': 'SocialDistancing', 'beta_mult': 0.6, 'gamma_mult': 1.0, 'vax_rate': 0.0},
    2: {'name': 'NoTravelRestriction', 'beta_mult': 1.0, 'gamma_mult': 1.0, 'vax_rate': 0.0},
    3: {'name': 'CompleteTravelLockdown', 'beta_mult': 0.3, 'gamma_mult': 1.0, 'vax_rate': 0.0},
    4: {'name': 'NormalHealthcare', 'beta_mult': 1.0, 'gamma_mult': 1.0, 'vax_rate': 0.0},
    5: {'name': 'EmergencyHealthcare', 'beta_mult': 1.0, 'gamma_mult': 2.0, 'vax_rate': 0.0},
    6: {'name': 'NoVaccination', 'beta_mult': 1.0, 'gamma_mult': 1.0, 'vax_rate': 0.0},
    7: {'name': 'TargetedVaccination', 'beta_mult': 1.0, 'gamma_mult': 1.0, 'vax_rate': 0.02},
    8: {'name': 'MassVaccination', 'beta_mult': 1.0, 'gamma_mult': 1.0, 'vax_rate': 0.05}
}

class EpidemicAgent:
    def __init__(self, data, agent_id):
        self.agent_id = agent_id
        self.location = data['location']

        # Epidemic state
        self.susceptible = float(data['susceptible'])
        self.infected = float(data['infected'])
        self.recovered = float(data['recovered'])
        self.deaths = float(data['deaths'])
        self.hospital_capacity = float(data['hospital_capacity'])
        self.vaccinated = float(data['vaccinated'])
        self.airports = float(data['airports'])

        # Track history for debugging
        self.infection_history = [self.infected]
        self.action_history = []

    def apply_action(self, action_idx):
        # Record action
        self.action_history.append(action_idx)

        # Get action effects
        effects = ACTION_EFFECTS.get(action_idx, ACTION_EFFECTS[0])

        # Base SIR parameters
        base_beta = 0.3   # Base transmission rate
        base_gamma = 0.1  # Base recovery rate

        # Apply action modifiers
        beta = base_beta * effects['beta_mult']
        gamma = base_gamma * effects['gamma_mult']
        vax_rate = effects['vax_rate']

        # Apply vaccination first
        if vax_rate > 0:
            vax_amount = min(self.susceptible * vax_rate, self.susceptible)
            self.vaccinated += vax_amount
            self.susceptible -= vax_amount

        # SIR dynamics
        total_pop = max(1, self.susceptible + self.infected + self.recovered)

        # New infections
        new_infected = beta * (self.susceptible * self.infected) / total_pop

        # Recoveries
        new_recovered = gamma * self.infected

        # Deaths (higher rate if hospitals overwhelmed)
        death_rate = 0.05 if self.infected > self.hospital_capacity else 0.01
        new_deaths = death_rate * self.infected

        # Update state
        self.susceptible = max(0, self.susceptible - new_infected)
        self.infected = max(0, self.infected + new_infected - new_recovered - new_deaths)
        self.recovered += new_recovered
        self.deaths += new_deaths

        # Track
        self.infection_history.append(self.infected)

    def calculate_reward(self):
        # Calculate metrics
        total_pop = max(1, self.susceptible + self.infected + self.recovered)
        infection_rate = self.infected / total_pop
        death_rate = self.deaths / total_pop
        vax_rate = self.vaccinated / total_pop
        hospital_util = self.infected / max(1, self.hospital_capacity)

        # Reward calculation
        reward = 0.0

        # Main objective: minimize infections and deaths
        reward -= infection_rate * 100.0    # -0 to -100
        reward -= death_rate * 500.0        # Heavy penalty for deaths

        # Reward vaccination progress
        reward += vax_rate * 50.0           # +0 to +50

        # Huge penalty for overwhelming hospitals
        if hospital_util > 1.0:
            reward -= (hospital_util - 1.0) * 200.0

        # Small penalty for using hospital resources
        reward -= hospital_util * 5.0

        # Bonus for controlling outbreak (low infections)
        if infection_rate < 0.01:  # Less than 1% infected
            reward += 20.0

        # Clamp to reasonable range
        reward = max(-100.0, min(100.0, reward))

        return reward

    def get_observation(self):
        return [
            self.susceptible / 1000000.0,
            self.infected / 10000.0,
            self.recovered / 10000.0,
            self.deaths / 1000.0,
            self.hospital_capacity / 20000.0,
            self.vaccinated / 1000000.0,
            self.airports / 10.0
        ]

# Initialize agents
agents = [EpidemicAgent(data, i) for i, data in enumerate(initial_data)]

# Print initial state
print("\\nInitial epidemic state:")
for agent in agents:
    print(f"  {agent.location}: I={int(agent.infected)}, S={int(agent.susceptible)}, R={int(agent.recovered)}")

row_data = json.loads('$jsonString')

def epidemic_obs_from_spark(env, agent):
    agent_id = int(agent.name.split("_")[1])

    # Initialize epidemic state on first call
    if not hasattr(env, 'epidemic_state'):
        env.epidemic_state = {}
        env.action_history = {}
        for idx, obs_values in enumerate(row_data):
            env.epidemic_state[idx] = {
                'susceptible': float(obs_values[0]),
                'infected': float(obs_values[1]),
                'recovered': float(obs_values[2]),
                'deaths': float(obs_values[3]),
                'hospital_capacity': float(obs_values[5]),
                'vaccinated': float(obs_values[8]),
                'airports': len(obs_values[7]),
            }
            env.action_history[idx] = []

    if agent_id >= len(row_data):
        default_obs = torch.zeros(7, dtype=torch.float32, device=env.world.device)
        agent.obs = default_obs
        return default_obs.unsqueeze(0)

    state = env.epidemic_state[agent_id]

    # Get the action from the agent object (VMAS sets this)
    if hasattr(agent, 'action') and agent.action is not None:
        try:
            action_idx = int(agent.action.item() if hasattr(agent.action, 'item') else agent.action)
            env.action_history[agent_id].append(action_idx)

            # Base transmission rate
            beta = 0.3
            gamma = 0.1

            # Action effects on transmission
            if action_idx == 1:  # SocialDistancing
                beta *= 0.6
            elif action_idx == 3:  # CompleteTravelLockdown
                beta *= 0.3
            elif action_idx == 8:  # MassVaccination
                vax_amt = min(state['susceptible'] * 0.05, state['susceptible'])
                state['vaccinated'] += vax_amt
                state['susceptible'] -= vax_amt
            elif action_idx == 7:  # TargetedVaccination
                vax_amt = min(state['susceptible'] * 0.02, state['susceptible'])
                state['vaccinated'] += vax_amt
                state['susceptible'] -= vax_amt
            elif action_idx == 5:  # EmergencyHealthcareMobilization
                gamma *= 2.0

            # SIR model dynamics
            total_pop = max(1, state['susceptible'] + state['infected'] + state['recovered'])
            new_infections = beta * state['susceptible'] * state['infected'] / total_pop
            new_recoveries = gamma * state['infected']

            # Death rate increases if hospitals overwhelmed
            death_rate = 0.01 if state['infected'] <= state['hospital_capacity'] else 0.08
            new_deaths = death_rate * state['infected']

            # Update state
            state['susceptible'] = max(0, state['susceptible'] - new_infections)
            state['infected'] = max(0, state['infected'] + new_infections - new_recoveries - new_deaths)
            state['recovered'] += new_recoveries
            state['deaths'] += new_deaths

        except Exception as e:
            print(f"Error processing action for agent {agent_id}: {e}")

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

    agent.obs = obs
    return obs.unsqueeze(0)

def epidemic_rf(env, agent):
    import torch
    agent_id = int(agent.name.split("_")[1])

    if not hasattr(env, 'epidemic_state') or agent_id not in env.epidemic_state:
        return torch.tensor(0.0, dtype=torch.float32, device=env.world.device)

    state = env.epidemic_state[agent_id]

    # Calculate reward based on epidemic metrics
    total_pop = max(1, state['susceptible'] + state['infected'] + state['recovered'])
    infection_rate = state['infected'] / total_pop
    death_rate = state['deaths'] / total_pop
    vax_rate = state['vaccinated'] / total_pop
    hospital_util = state['infected'] / max(1, state['hospital_capacity'])

    reward = 0.0
    reward -= infection_rate * 100.0  # Penalize infections
    reward -= death_rate * 500.0      # Heavily penalize deaths
    reward += vax_rate * 50.0         # Reward vaccination

    if hospital_util > 1.0:
        reward -= (hospital_util - 1.0) * 200.0  # Huge penalty for hospital overflow

    reward = max(-100.0, min(100.0, reward))

    return torch.tensor(reward, dtype=torch.float32, device=env.world.device)
"""

  CPythonInterpreter.execManyLines(epidemicSimulationCode)

  val obsLambda = py.Dynamic.global.epidemic_obs_from_spark
  val rfLambda = py.Dynamic.global.epidemic_rf

  // ============================================================================
  // VMAS ENVIRONMENT SETUP
  // ============================================================================

  WANDBLogger.init()

  CPythonInterpreter.execManyLines("""
import sys, os
paths = ["./src/main/resources", "./build/resources/main", "./src/main/scala/resources"]
for p in paths:
    abs_p = os.path.abspath(p)
    if os.path.isdir(p) and abs_p not in sys.path:
        sys.path.append(abs_p)
""")

  val scenario = py.module("AbstractEnv").Scenario(rfLambda, obsLambda)

  val envSettings = VmasSettings(
    scenario = scenario,
    nEnv = 1,
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

  val descriptor = VmasStateDescriptor(hasPosition = false, hasVelocity = false, extraDimension = 7)
  VMASEpidemicState.setDescriptor(descriptor)

  // ============================================================================
  // LEARNING SYSTEM
  // ============================================================================

  val where = "./epidemic_networks"

  val epidemicSystem = CTDELearningSystem {
    rewardFunction { DSLRewardFunction()  }
    actionSpace { RealEpidemicAction.toSeq }
    dataset { ReplayBuffer[State, Action](10000) }
    agents { nAgents }
    learningConfiguration {
      LearningConfiguration(
        dqnFactory = new EpidemicNNFactory(7, RealEpidemicAction.toSeq),
        snapshotPath = where
      )
    }
    environment { "vmas.VmasEpidemicEnvironment" }
  }(ExecutionContext.global, VMASEpidemicState.encoding)

  // ============================================================================
  // TRAINING
  // ============================================================================

  println("\nStarting training...")

  for (epoch <- 1 to nEpochs) {
    if (epoch % 20 == 0) {
      println(s"Epoch $epoch/$nEpochs")
    }
    epidemicSystem.learn(1, nSteps)
  }

  println("\nTraining completed!")

  // Print final epidemic states
  CPythonInterpreter.execManyLines("""
print("\\nFinal epidemic state:")
for agent in agents:
    print(f"  {agent.location}: I={int(agent.infected)}, S={int(agent.susceptible)}, R={int(agent.recovered)}, D={int(agent.deaths)}")
    print(f"    Vaccinated: {int(agent.vaccinated)}, Last 5 actions: {agent.action_history[-5:]}")
""")

  // ============================================================================
  // VALIDATION
  // ============================================================================

  println("\n" + "="*80)
  println("MODEL VALIDATION")
  println("="*80)

  try {
    println("Executing validation code...")

    val validationCode = """
import glob, os, torch, torch.nn as nn

print("Step 1: Looking for checkpoints...")
files = glob.glob("epidemic_networks/*")
print(f"Found {len(files)} checkpoint files")

if not files:
    print("ERROR: No checkpoints found")
    print("Check if epidemic_networks directory exists")
else:
    print(f"Step 2: Loading latest checkpoint...")
    latest = max(files, key=os.path.getctime)
    print(f"Loading: {latest}")

    try:
        checkpoint = torch.load(latest, map_location="cpu")
        print("Checkpoint loaded successfully")

        model = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 9)
        )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            print("Loaded from state_dict")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded checkpoint directly")

        model.eval()
        print("Model ready for validation")

        # Test on one simple input first
        test_obs = torch.tensor([[0.99, 0.01, 0.0, 0.0, 0.5, 0.05, 0.2]], dtype=torch.float32)
        test_output = model(test_obs)
        print(f"Test output shape: {test_output.shape}, values: {test_output.detach().numpy()}")

        print("\\nValidation complete!")

    except Exception as e:
        print(f"Error during model loading: {e}")
        import traceback
        traceback.print_exc()
"""

    CPythonInterpreter.execManyLines(validationCode)
    println("Validation execution completed")

  } catch {
    case e: Exception =>
      println(s"SCALA ERROR during validation: ${e.getMessage}")
      e.printStackTrace()
  }

  println("="*80)
  spark.stop()
  println("Complete!")
}

case class DSLRewardFunction() extends RewardFunction {
  override def compute(currentState: State, action: Action, newState: State): Double = {
    // DEBUG: Check if we ever get called
    println(s"DSLRewardFunction.compute called - this should appear during training!")

    val reward = RewardFunctionDSL.rf match {
      case Some(r) =>
        println("Using RewardFunctionDSL.rf")
        r.compute(currentState, action, newState)
      case None =>
        println("WARNING: RewardFunctionDSL.rf is None - returning 0.0")
        0.0
    }

    // Always print the reward value
    println(s"DSLRewardFunction returning: $reward")
    reward
  }
}

// Reward function that attempts to get reward from VMAS environment
/*case class PythonRewardFunction() extends RewardFunction {
  override def compute(currentState: State, action: Action, newState: State): Double = {
    // Try to extract reward that VMAS calculated
    newState match {
      case s: VMASEpidemicState =>
        // Check if VMAS stored the reward in the state somehow
        // This is a hack - ideally scarlib should integrate properly with VMAS
        try {
          // Attempt to get the last reward from Python
          val rewardTensor = py.Dynamic.global.last_reward
          if (rewardTensor != null) {
            rewardTensor.item().as[Double]
          } else {
            println("WARNING: last_reward is null")
            0.0
          }
        } catch {
          case e: Exception =>
            // Fall back to calculating reward ourselves
            calculateRewardFromState(s)
        }
      case _ => 0.0
    }
  }

  private def calculateRewardFromState(state: VMASEpidemicState): Double = {
    // Manual reward calculation as fallback
    // This should match your Python reward function logic
    try {
      val obs = state.tensor
      // Extract values from observation tensor
      // obs[1] is infected rate (normalized)
      val infectedNorm = obs.bracketAccess(1).as[Double]
      val infected = infectedNorm * 10000.0

      // Simple penalty for infections
      val reward = -infected / 100.0
      Math.max(-100.0, Math.min(100.0, reward))
    } catch {
      case e: Exception =>
        println(s"Error calculating reward: ${e.getMessage}")
        0.0
    }
  }
}*/