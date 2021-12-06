
import Foundation
import CreateML

// Initializing the properly labeled training data from Resources folder.
let trainingDataPath = Bundle.main.path(forResource: "Cat", ofType: nil, inDirectory: "Training Data")!
let trainingData = MLImageClassifier.DataSource.labeledDirectories(at: URL(fileURLWithPath: trainingDataPath))

// Initializing the classifier with a training data.
let classifier = try! MLImageClassifier(trainingData: trainingData)


// Evaluating training & validation accuracies.
let trainingAccuracy = (1.0 - classifier.trainingMetrics.classificationError) * 100 // Result: 100%
let validationAccuracy = (1.0 - classifier.validationMetrics.classificationError) * 100 // Result: 100%

// Initializing the properly labeled testing data from Resources folder.
let testingDataPath = Bundle.main.path(forResource: "Cat", ofType: nil, inDirectory: "Testing Data")!
let testingData = MLImageClassifier.DataSource.labeledDirectories(at: URL(fileURLWithPath: testingDataPath))

// Counting the testing evaluation.
let evaluationMetrics = classifier.evaluation(on: testingData)
let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100  // Result: 97.22%

// Confusion matrix in order to see which labels were classified wrongly.
let confusionMatrix = evaluationMetrics.confusion
print("Confusion matrix: \(confusionMatrix)")

// Metadata for saving the model.
let metadata = MLModelMetadata(author: "Author",
                               shortDescription: "A model trained to distinguish pets.",
                               version: "1.0")

// Saving the model. Remember to update the path.
try classifier.write(to: URL(fileURLWithPath: "Path where you would like to save the model"),
                     metadata: metadata)



//let builder = MLImageClassifierBuilder()

