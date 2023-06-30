//NOTE:This script works only for QuPath v3.
//This script is a modified version of example script written in QuPath docs for v3 (https://qupath.readthedocs.io/en/0.3/docs/advanced/stardist.html).

selectAnnotations();
import qupath.ext.stardist.StarDist2D
var entry = getProjectEntry()
var name = entry.getImageName() + ".txt"

// Specify the model file (you will need to change this!)
var pathModel = '/path/to/he_heavy_augment.pb' 
var stardist = StarDist2D.builder(pathModel)
        .threshold(0.5)              // Probability (detection) threshold
        .normalizePercentiles(1, 99) // Percentile normalization
        .pixelSize(0.3)              // Resolution for detection
        .tileSize(1024)              // Specify width & height of the tile used for prediction
        .cellExpansion(5.0)          // Approximate cells based upon nucleus expansion
        .cellConstrainScale(2.0)     // Constrain cell expansion using nucleus size
        .ignoreCellOverlaps(false)   // Set to true if you don't care if cells expand into one another
        .measureShape()              // Add shape measurements
        .measureIntensity()          // Add cell measurements (in all compartments)
        .includeProbability(true)    // Add probability as a measurement (enables later filtering)
        .nThreads(4)                 // Limit the number of threads used for (possibly parallel) processing
        .simplify(1)                 // Control how polygons are 'simplified' to remove unnecessary vertices
        .doLog()                     // Use this to log a bit more information while running the script
        .build()

// Run detection for the selected objects
var imageData = getCurrentImageData()
var pathObjects = getSelectedObjects()
if (pathObjects.isEmpty()) {
    Dialogs.showErrorMessage("StarDist", "Please select a parent object!")
    return
}
stardist.detectObjects(imageData, pathObjects)
saveDetectionMeasurements("/path/to/directory/" + name)
println 'Done!'


