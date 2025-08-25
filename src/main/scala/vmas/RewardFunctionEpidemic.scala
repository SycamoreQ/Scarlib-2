package vmas

import it.unibo.scarlib.core.model._
import me.shadaj.scalapy.interpreter.CPythonInterpreter
import me.shadaj.scalapy.py

import scala.language.implicitConversions

object RewardFunctionEpidemic {

  CPythonInterpreter.execManyLines("import torch")

  private var pythonCode = ""
  private var rf: Option[RewardFunction] = None

  abstract class RewardFunctionStep(param: RewardFunctionStepParam) {
    def compute()(implicit currentState: State, action: Action, newState: State): Any
  }

  def rewardFunctionStep(init: => RewardFunctionStep): Unit = {
    rf = Option(new RewardFunction {
      override def compute(currentState: State, action: Action, newState: State): Double = init.compute()(currentState, action, newState).asInstanceOf[Double]
    })
  }

  trait RewardFunctionStepParam

  case object CurrentState extends RewardFunctionStepParam

  case object Action extends RewardFunctionStepParam

  case object NewState extends RewardFunctionStepParam


  case class Tensor(x: py.Dynamic, stringed: String)

  object Tensor {
    def apply(x: Any): Tensor = new Tensor(py.eval(s"torch.tensor($x)"), s"torch.tensor($x)")
  }

  case class Lambda(x: py.Dynamic, stringed: String)

  object Lambda {
    def apply(x: Any): Lambda = new Lambda(py.eval("lambda " + x.toString), "lambda " + x.toString)
  }

  implicit def tensorToPyDynamic(x: Tensor): (py.Dynamic, String) = (x.x, x.stringed)

  implicit def doubleToPyDynamic(x: Double): (py.Dynamic, String) = (py.eval(x.toString), x.toString)

  implicit def lambdaToPyDynamic(x: Lambda): (py.Dynamic, String) = (x.x, x.stringed)

  case class InfectionPenalty(x: (py.Dynamic, String) , param : RewardFunctionStepParam) extends RewardFunctionStep(param) {
    override def compute()(implicit currentState: State, action: Action, newState: State): py.Dynamic = {

      val state = param match {
        case CurrentState => currentState
        case NewState => newState
      }

      state  match {
        case s: VMASEpidemicState =>
          val infRate = s.getInfectionRate
          s.tensor + x._1 * infRate * 100

        case _ => return py.eval("0.0")
      }
    }

    override def toString: String = {
      val state = param match {
        case CurrentState => "agent.currentstate"
        case NewState => "agent.newstate"
      }
      s"($state + ${x._2})"
    }
  }

  case class hospitalUtilization(x: (py.Dynamic, String) , param : RewardFunctionStepParam) extends RewardFunctionStep(param) {
    override def compute()(implicit currentState: State, action: Action, newState: State): py.Dynamic = {

      val state = param match {
        case CurrentState => currentState
        case NewState => newState
      }

      state  match {
        case s: VMASEpidemicState =>
          val hUtil = s.getHospitalUtilization

          if (hUtil > 10000) {
            s.tensor + x._1 * hUtil * 100
          }

          if (hUtil > 1000){
            s.tensor + x._1*hUtil*10
          }

          else{
            s.tensor + x._1 *hUtil*2
          }

        case _ => return py.eval("0.0")
      }
    }

    override def toString: String = {
      val state = param match {
        case CurrentState => "agent.currentstate"
        case NewState => "agent.newstate"
      }
      s"($state + ${x._2})"
    }
  }

  case class VaccinationDrive(x: (py.Dynamic, String) , param : RewardFunctionStepParam) extends RewardFunctionStep(param) {
    override def compute()(implicit currentState: State, action: Action, newState: State): py.Dynamic = {

      val state = param match {
        case CurrentState => currentState
        case NewState => newState
      }

      state match {
        case s: VMASEpidemicState =>
          val vaccinated = s.getVaccinationRate
          val infected = s.epidemicData.map(_.infected).sum
          val incomingTravellers = s.epidemicData.map(_.incomingTravelers.values.sum).getOrElse(0)

          if (vaccinated > infected) {
            s.tensor - x._1 * vaccinated * 100
          }

          if (vaccinated > incomingTravellers){
            s.tensor + x._1*vaccinated*10
          }

          else{
            s.tensor + x._1 *vaccinated*2
          }

        case _ => py.eval("0.0")
      }
    }

    override def toString: String = {
      val state = param match {
        case CurrentState => "agent.currentstate"
        case NewState => "agent.newstate"
      }
      s"($state + ${x._2})"
    }
  }

  case class airportFunc(x: (py.Dynamic, String) , diseaseCountry: String , targetCountry : Seq[String] ,  param : RewardFunctionStepParam) extends RewardFunctionStep(param) {
    override def compute()(implicit currentState: State, action: Action, newState: State): py.Dynamic = {

      val state = param match {
        case CurrentState => currentState
        case NewState => newState
      }

      state match {
        case s: VMASEpidemicState =>
          // Get Option[Seq[(String, Int)]]
          val maybeConnected: Option[Seq[(String, Int)]] = s.epidemicData.map(_.getMostConnectedCountries(diseaseCountry, targetCountry))
          // Get Option[(String, Int)] for the most connected country, or None
          val mostConnected: Option[(String, Int)] = maybeConnected.flatMap(_.headOption)
          mostConnected match {
            case Some((country, volume)) =>
              // Apply a heavier penalty for the most connected country
              s.tensor - x._1 * s.getInfectionRate * 100
            case None =>
              // No penalty if there are no connections
              s.tensor
          }
        case _ => py.eval("0.0")
      }

    }

    override def toString: String = {
      val state = param match {
        case CurrentState => "agent.currentstate"
        case NewState => "agent.newstate"
      }
      s"($state + ${x._2})"
    }
  }

  case class AddRoot(x: (py.Dynamic, String), param: RewardFunctionStepParam) extends RewardFunctionStep(param) {
    override def compute()(implicit currentState: State, action: Action, newState: State): Any = {
      val state = param match {
        case CurrentState => currentState
        case NewState => newState
      }
      state match {
        case state: VMASState =>
          state.tensor + x._1
        case _ =>
          py.eval("0.0")
      }
    }

    override def toString: String = {
      val state = param match {
        case CurrentState =>  "agent.currentState"
        case NewState => "agent.newState"
      }
      s"($state + ${x._2})"
    }
  }

  case class Add(x: (py.Dynamic, String), y: RewardFunctionStep) extends RewardFunctionStep(CurrentState) {
    override def compute()(implicit currentState: State, action: Action, newState: State): Any = {
      x._1 + y.compute()(currentState, action, newState).asInstanceOf[py.Dynamic]
    }

    override def toString: String = s"(${x._2} + $y)"

  }

  case class Sub(x: py.Dynamic, param: RewardFunctionStepParam) extends RewardFunctionStep(param) {
    override def compute()(implicit currentState: State, action: Action, newState: State): Any = {
      val state = param match {
        case CurrentState => currentState
        case NewState => newState
      }
      state match {
        case state: VMASState =>
          state.tensor - x
        case _ =>
          py.eval("0.0")
      }
    }
  }

  case class Mul(x: py.Dynamic, param: RewardFunctionStepParam) extends RewardFunctionStep(param) {
    override def compute()(implicit currentState: State, action: Action, newState: State): Any = {
      val state = param match {
        case CurrentState => currentState
        case NewState => newState
      }
      state match {
        case state: VMASState =>
          state.tensor * x
        case _ =>
          py.eval("0.0")
      }
    }
  }

  case class Div(x: py.Dynamic, param: RewardFunctionStepParam) extends RewardFunctionStep(param) {
    override def compute()(implicit currentState: State, action: Action, newState: State): Any = {
      val state = param match {
        case CurrentState => currentState
        case NewState => newState
      }
      state match {
        case state: VMASState =>
          state.tensor / x
        case _ =>
          py.eval("0.0")
      }
    }
  }

  case class AddTwoStep(x: RewardFunctionStep, y: RewardFunctionStep) extends RewardFunctionStep(CurrentState) {
    override def compute()(implicit currentState: State, action: Action, newState: State): Any =
      (x.compute()(currentState, action, newState)).asInstanceOf[py.Dynamic] + (y.compute()(currentState, action, newState)).asInstanceOf[py.Dynamic]

    override def toString: String = s"($x + $y)"
  }

  case class Map(x: RewardFunctionStep, lambda: (py.Dynamic, String)) extends RewardFunctionStep(CurrentState) {

    override def toString: String = s"(${lambda._2})($x)"

    override def compute()(implicit currentState: State, action: Action, newState: State): Any = {
      val returnValue = lambda._1(x.compute()(currentState, action, newState).asInstanceOf[py.Dynamic])
      returnValue
    }
  }

  case class Reduce(x: RewardFunctionStep, lambda: (py.Dynamic, String)) extends RewardFunctionStep(CurrentState) {
    override def compute()(implicit currentState: State, action: Action, newState: State): Any = {
      val returnValue = lambda._1(x.compute()(currentState, action, newState).asInstanceOf[py.Dynamic]).asInstanceOf[Double]
      returnValue
    }

    override def toString: String = s"(${lambda._2})($x)"
  }

  implicit class AddOps(x: RewardFunctionStep) {
    def +(y: (py.Dynamic, String)): RewardFunctionStep = Add(y, x)

  }

  implicit class MapOps(x: RewardFunctionStep) {
    def -->(y: (py.Dynamic, String)): RewardFunctionStep = Map(x, y)
  }

  implicit class ReduceOps(x: RewardFunctionStep) {
    def >>(lambda: (py.Dynamic, String)): RewardFunctionStep = Reduce(x, lambda)
  }

  implicit class AddTwoStepOps(x: RewardFunctionStep) {
    def ++(y: RewardFunctionStep): RewardFunctionStep = AddTwoStep(x, y)
  }
}