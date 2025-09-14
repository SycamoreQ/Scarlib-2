package vmas

import scarlib.neuralnetwork.{DQNAbstractFactory, SimpleSequentialDQN}
import me.shadaj.scalapy.py

class NNFactory(stateDescriptor: VmasStateDescriptor, actionsSpace: Seq[VMASAction]) extends DQNAbstractFactory[py.Dynamic] {

        override def createNN(): py.Dynamic = {
            SimpleSequentialDQN(stateDescriptor.getSize, 64, actionsSpace.size)
        }

}

class EpidemicNNFactory(input : Int = 4  , actionSpace : Seq[VMASAction]) extends DQNAbstractFactory[py.Dynamic]{

  override def createNN(): py.Dynamic = {
    SimpleSequentialDQN(input , 64 , actionSpace.size)
  }
}

