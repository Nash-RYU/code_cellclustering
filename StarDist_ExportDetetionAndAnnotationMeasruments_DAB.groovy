selectAnnotations();

import qupath.tensorflow.stardist.StarDist2D


def entry = getProjectEntry()
def name = entry.getImageName() + ".txt"
// Specify the model directory (you will need to change this!)
def pathModel = "/home/nash/qupath_scripts/he_heavy_augment"

def stardist = StarDist2D.builder(pathModel)
        .threshold(0.5)              // Probability (detection) threshold
        .normalizePercentiles(1, 99) // Percentile normalization
        .pixelSize(0.3)              // Resolution for detection
        .cellExpansion(10.0)          // Approximate cells based upon nucleus expansion
        .cellConstrainScale(2.0)     // Constrain cell expansion using nucleus size
        .measureShape()              // Add shape measurements
        .measureIntensity()          // Add cell measurements (in all compartments)
        .includeProbability(true)    // Add probability as a measurement (enables later filtering)
        .build()

// Run detection for the selected objects
def imageData = getCurrentImageData()
def pathObjects = getSelectedObjects()
if (pathObjects.isEmpty()) {
    Dialogs.showErrorMessage("StarDist", "Please select a parent object!")
    return
}
stardist.detectObjects(imageData, pathObjects)

saveDetectionMeasurements("/media/nash/Transcend/QuPathProject/DetectionMeasurement_SOX9/Progression_exome/" + name)

setCellIntensityClassifications('DAB: Nucleus: Mean', 0.1, 0.3, 0.5)
saveAnnotationMeasurements("/media/nash/Transcend/QuPathProject/AnnotationMeasurement_SOX9/Progression_exome/" + name)



println 'Done!'
