package vmas

import scarlib.model.{AutodiffDevice, State}
import scarlib.neuralnetwork.NeuralNetworkEncoding
import vmas.* 
import me.shadaj.scalapy.py
import me.shadaj.scalapy.readwrite.Reader.{doubleReader, _}



object VMASState{
    def apply(array: py.Dynamic): VMASState = new VMASState(py.module("torch").tensor(array).to(AutodiffDevice()))
    private var stateDescriptor: Option[VmasStateDescriptor] = None
    def setDescriptor(descriptor: VmasStateDescriptor): Unit = stateDescriptor = Some(descriptor)

    implicit val encoding: NeuralNetworkEncoding[State] = new NeuralNetworkEncoding[State] {

        /** Gets the number of elements in the state */
        override def elements(): Int = stateDescriptor match {
            case Some(descriptor) => descriptor.getSize
            case None => throw new Exception("State descriptor not set")
        }

        /** Converts the state into a format usable by the neural network */
        override def toSeq(element: State): Seq[Double] = {
          val pythonList = element.asInstanceOf[VMASState].tensor.flatten().tolist
          val length = pythonList.__len__().as[Int]
          (0 until length).map(i => pythonList.__getitem__(i).as[Double]).toSeq
        }

    }

}

class VMASState(val tensor: py.Dynamic) extends State {

  /** Checks if the state is empty */
    override def isEmpty(): Boolean = false

}



