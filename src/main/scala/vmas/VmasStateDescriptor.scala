package vmas

import scala.math.*
import me.shadaj.scalapy.py
import org.eclipse.core.internal.registry.ExtensionRegistry

case class VmasStateDescriptor(hasPosition: Boolean = true, hasVelocity: Boolean = true, positionDimensions: Int = 2,
                               velocityDimensions: Int = 2, lidarDimension: Int = 1, lidars: Seq[Int] = Seq.empty, extraDimension: Int = 0 ,
                               hasTensorSize: Boolean = true) {
    def getSize: Int = {
        var totalSize = extraDimension
        if (hasPosition) totalSize += positionDimensions
        if (hasVelocity) totalSize += velocityDimensions
        if (lidars.nonEmpty) totalSize += lidars.sum * lidarDimension
        return totalSize
    }
    
    def getTensorSize : Int =  {
      var TensorSize = 0 
      if (hasTensorSize) TensorSize += 1
      TensorSize
    }
}


