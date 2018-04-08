using System;
using System.Drawing;
using System.Linq;
using System.Data;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Accord.Statistics.Models.Regression.Linear;
using Accord.Statistics.Analysis;
using Accord.IO;
using Accord.Math;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Kernels;
using Accord.Controls;
using Accord.MachineLearning;
using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.MachineLearning.DecisionTrees;
using Accord.Math.Optimization;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.Statistics.Filters;
using Accord.MachineLearning.Bayes;
using Accord.Statistics.Distributions.Univariate;
using Accord.Statistics.Distributions.Fitting;
using Accord.MachineLearning.Performance;
using Accord.Statistics;

namespace ML
{
    class Program
    {
        static void Main(string[] args)
        {
            ImageToCSV itc = new ImageToCSV();
            Bitmap[] bm = itc.InitImages(1);
            itc.GetPixelImages(bm);
            itc.SaveCSV();

           // DataTable data_train = new CsvReader(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\train.csv", true).ToTable();
           // DataTable data_test = new CsvReader(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\test.csv", true).ToTable();

           // int[] trainOutputs = data_train.Columns["label"].ToArray<int>();
           // data_train.Columns.Remove("label");
           // double[][] trainInputs = data_train.ToJagged<double>();

           //// int[] testOutputs = data_test.Columns["label"].ToArray<int>();
           // //data_test.Columns.Remove("label");
           // double[][] testInputs = data_test.ToJagged<double>();



           // var knn = new KNN(trainInputs, trainOutputs, 4);
           // var machine_knn = knn.MachineLearning();
           // int[] predicted_knn = machine_knn.Decide(testInputs);

           // PrintResultsСlassifier show_knn = new PrintResultsСlassifier(machine_knn, testInputs);//, testOutputs);

           // show_knn.PrintProbabilities();





            //show_knn.PrintPredicted(predicted_knn);
            //show_knn.PrintAccuracy();

            //Console.WriteLine("BEGIN NaiveBayes");

            //var nb = new NB(trainInputs, trainOutputs);
            //var machine_nb = nb.MachineLearning();
            //int[] predicted_nb = machine_nb.Decide(testInputs);

            //PrintResultsСlassifier show_nb = new PrintResultsСlassifier(machine_nb, testInputs, testOutputs);
            //show_nb.PrintPredicted(predicted_knn);
            //show_nb.PrintAccuracy();
            //show_nb.PrintProbabilities();

            //double loss = new ZeroOneLoss(testOutputs).Loss(predicted_knn);
            ////loss , machine_knn <double[]>


            //var cv = CrossValidation.Create(

            //    k: 10, // We will be using 10-fold cross validation

            //    // First we define the learning algorithm:
            //    learner: (p) => new NaiveBayesLearning(),

            //    // Now we have to specify how the n.b. performance should be measured:
            //    loss: (actual, expected) => new ZeroOneLoss(expected).Loss(actual),

            //    // This function can be used to perform any special
            //    // operations before the actual learning is done, but
            //    // here we will just leave it as simple as it can be:
            //    fit: (NaiveBayesLearning teacher, double[][] x, int[] y, int[] w) => teacher.Learn(x, y, w),

            //    // Finally, we have to pass the input and output data
            //    // that will be used in cross-validation. 
            //    x: testInputs, y: testOutputs
            //);

            //// After the cross-validation object has been created,
            //// we can call its .Learn method with the input and 
            //// output data that will be partitioned into the folds:
            //var result = cv.Learn(testInputs, testOutputs);

            //// We can grab some information about the problem:
            //int numberOfSamples = result.NumberOfSamples; // should be 15
            //int numberOfInputs = result.NumberOfInputs;   // should be 4
            //int numberOfOutputs = result.NumberOfOutputs; // should be 3

            //double trainingError = result.Training.Mean; // should be 0
            //double validationError = result.Validation.Mean; // should be 0.15 (+/- var. 0.11388888888888887)

            //// If desired, compute an aggregate confusion matrix for the validation sets:
            //GeneralConfusionMatrix gcm = result.ToConfusionMatrix(inputs, outputs);

            //Console.ReadLine();
        }
    }
}
