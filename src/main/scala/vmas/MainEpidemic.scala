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

object MainEpidemic extends App {

  // Environment parameters
  val nAgents = 5  // Number of countries/regions in epidemic simulation
  val nSteps = 100
  val nEpochs = 150

  // Initialize Python environment
  CPythonInterpreter.execManyLines("import torch")
  CPythonInterpreter.execManyLines("import numpy as np")

  // Define epidemic-specific constants
  val diseaseOrigin = "China"
  val targetCountries = Seq("Italy", "USA", "Germany", "France")

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

  val descriptor = VmasStateDescriptor()
  VMASEpidemicState.setDescriptor(descriptor)


  // Define epidemic reward function in Python using your DSL components
  // This translates your Scala DSL to executable Python code
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

  // Define epidemic observation function (similar to Main.scala obs function)
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

  val obsLambda = py.Dynamic.global.epidemic_obs

  // Initialize logging
  WANDBLogger.init()

  // Ensure Python can import local modules like AbstractEnv.py
  CPythonInterpreter.execManyLines(
    """import sys, os; [sys.path.append(os.path.abspath(p)) for p in ["./src/main/resources","./build/resources/main","./src/main/scala/resources"] if os.path.isdir(p) and os.path.abspath(p) not in sys.path]"""
  )


  // Now you can safely import AbstractEnv
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

  // Environment configuration (similar to Main.scala configuration)
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
        dqnFactory = new EpidemicNNFactory(1000, RealEpidemicAction.toSeq),
        snapshotPath = where
      )
    }

    environment {
      "vmas.VmasEpidemicEnvironment"
    }
  } (ExecutionContext.global, VMASEpidemicState.encoding)

  println("Starting epidemic simulation training...")
  epidemicSystem.learn(envSettings.nEpochs, envSettings.nSteps)
}


case class EmptyRewardFunctionEpidemic() extends RewardFunction{
  override def compute(currentState: State, action: Action, newState: State): Double = 0.0
}
