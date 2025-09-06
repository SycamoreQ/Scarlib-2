package vmas

import scarlib.model.{AutodiffDevice, State}
import scarlib.neuralnetwork.NeuralNetworkEncoding
import vmas.RewardFunctionEpidemic.Tensor
import vmas.VmasStateDescriptor
import me.shadaj.scalapy.py
import me.shadaj.scalapy.readwrite.Reader.doubleReader

import scala.math._



case class VMASEpidemicStateDescriptor(
                                        tensorSize: Int = 15,
                                        hasPopulationFeatures: Boolean = true,
                                        hasTravelFeatures: Boolean = true,
                                        hasHospitalFeatures: Boolean = true,
                                        hasTimeFeatures: Boolean = true
                                      ) {
  def getTensorSize: Int = tensorSize
}

case class EpidemicData(
                         susceptible: Int,
                         infected: Int,
                         recovered: Int,
                         deaths: Int,
                         exposed: Int = 0,
                         hospitalCapacity: Int,
                         location: String,
                         airports: List[String] = List.empty,
                         vaccinatedPopulation: Int = 0,
                         travelVolume: Int = 0,
                         currentDate: String = "",
                         previousInfected: Int = 0,
                         previousRecovered: Int = 0,
                         previousDeaths: Int = 0,
                         ageDistribution: Map[String, Int] = Map("0-18" -> 0, "19-64" -> 0, "65+" -> 0),
                         incomingTravelers: Map[String, Int] = Map.empty,
                         outgoingTravelers: Map[String, Int] = Map.empty,
                         airportTraffic: Map[String, Map[String, Int]] = Map.empty
                       ) {

  // Business logic methods belong here
  def getInfectionRate: Double = {
    if (infected > 0) {
      (recovered - previousRecovered).toDouble / infected
    } else 0.0
  }

  def getTotalPopulation: Int = susceptible + infected + recovered + deaths + exposed

  def getHospitalUtilization: Double = {
    val estimatedHospitalizations = (infected * 0.15).toInt
    if (hospitalCapacity > 0) {
      min(1.0, estimatedHospitalizations.toDouble / hospitalCapacity)
    } else 1.0
  }

  def getVaccinationRate: Double = {
    if (getTotalPopulation > 0) {
      vaccinatedPopulation.toDouble / getTotalPopulation
    } else 0.0
  }

  def getTravelVolumeTo(destination: String): Int = {
    outgoingTravelers.getOrElse(destination, 0)
  }

  def getAirportTravelVolume(airportCode: String, destination: String): Int = {
    airportTraffic.get(airportCode) match {
      case Some(destinations) => destinations.getOrElse(destination, 0)
      case None => 0
    }
  }

  def getBilateralVolume(c1: String, c2: String): (Int, Int) = {
    val c1_c2 = getTravelVolumeTo(c1)
    val c2_c1 = getTravelVolumeTo(c2)
    (c1_c2, c2_c1)
  }

  def getTotalTravelVolumeBetween(country1: String, country2: String): Int = {
    val (vol1to2, vol2to1) = getBilateralVolume(country1, country2)
    vol1to2 + vol2to1
  }

  def getMostConnectedCountries(
                                 targetCountry: String,
                                 allCountries: Seq[String],
                                 topN: Int = 5
                               ): Seq[(String, Int)] = {
    allCountries
      .filter(_ != targetCountry)
      .map(country => (country, getTotalTravelVolumeBetween(targetCountry, country)))
      .sortBy(_._2)(Ordering.Int.reverse)
      .take(topN)
  }

  def calculateTravelInfectionRisk(destinationCountry: String): Double = {
    val travelVolume = getTravelVolumeTo(destinationCountry)
    val originInfectionRate = getInfectionRate
    val destinationPopulation = getTotalPopulation

    if (destinationPopulation > 0 && travelVolume > 0) {
      (travelVolume * originInfectionRate) / destinationPopulation
    } else 0.0
  }
}


class VMASEpidemicState(
                         val tensor: py.Dynamic,
                         val epidemicData: Option[EpidemicData] = None
                       ) extends State {

  override def isEmpty(): Boolean = false

  def getInfectionRate: Double = {
    epidemicData.map(_.getInfectionRate).getOrElse(extractFromTensor("infection_rate"))
  }

  def getHospitalUtilization: Double = {
    epidemicData.map(_.getHospitalUtilization).getOrElse(extractFromTensor("hospital_utilization"))
  }

  def getTotalPopulation: Int = {
    epidemicData.map(_.getTotalPopulation).getOrElse(extractFromTensor("total_population").toInt)
  }

  def getVaccinationRate: Double = {
    epidemicData.map(_.getVaccinationRate).getOrElse(extractFromTensor("vaccination_rate"))
  }


  private def extractFromTensor(field: String): Double = {
    try {
      field match {
        case "infection_rate" => tensor.narrow(0, 0, 1).item().as[Double]
        case "hospital_utilization" => tensor.narrow(0, 1, 1).item().as[Double]
        case "total_population" => tensor.narrow(0, 2, 1).item().as[Double] * 1000000
        case "vaccination_rate" => tensor.narrow(0, 3, 1).item().as[Double]
        case _ => 0.0
      }
    } catch {
      case _: Exception => 0.0
    }
  }
}

object VMASEpidemicState {

  def apply(array: py.Dynamic): VMASEpidemicState =
    new VMASEpidemicState(py.module("torch").tensor(array).to(AutodiffDevice()))

  private var stateDescriptor: Option[VmasStateDescriptor] = None

  def setDescriptor(descriptor: VmasStateDescriptor): Unit = stateDescriptor = Some(descriptor)

  implicit val encoding: NeuralNetworkEncoding[State] = new NeuralNetworkEncoding[State] {

    override def elements(): Int = stateDescriptor match {
      case Some(descriptor) => descriptor.getSize
      case None => throw new Exception("State descriptor not set")
    }

    override def toSeq(element: State): Seq[Double] = {
      val pythonList = element.asInstanceOf[VMASEpidemicState].tensor.flatten().tolist()
      val length = pythonList.__len__().as[Int]
      (0 until length).map(i => pythonList.__getitem__(i).as[Double]).toSeq
    }
  }
}


object EpidemicDataEncoder {

  def encodeToTensor(data: EpidemicData): py.Dynamic = {
    val totalPop = data.getTotalPopulation.toDouble
    val normalizationFactor = if (totalPop > 0) totalPop else 1.0

    val features = Seq(
      data.susceptible.toDouble / normalizationFactor,
      data.infected.toDouble / normalizationFactor,
      data.recovered.toDouble / normalizationFactor,
      data.deaths.toDouble / normalizationFactor,
      data.exposed.toDouble / normalizationFactor,
      data.hospitalCapacity.toDouble / 100000.0,
      data.vaccinatedPopulation.toDouble / normalizationFactor,
      data.travelVolume.toDouble / 10000.0,
      data.airports.length.toDouble / 10.0,
      data.getInfectionRate,
      data.getHospitalUtilization,
      data.getVaccinationRate,
      data.incomingTravelers.values.sum.toDouble / 10000.0,
      data.outgoingTravelers.values.sum.toDouble / 10000.0,
      data.ageDistribution.values.sum.toDouble / normalizationFactor
    )

    features.asInstanceOf[py.Dynamic]
  }
}