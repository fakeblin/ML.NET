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
using ML.Classifier;

namespace ML
{
    class Program
    {
        static void Main(string[] args)
        {

            //double[][] trainInputs =
            //{
            //// The first two are from class 0
            //new double[] { -5, -2, -1 },
            //new double[] { -5, -5, -6 },

            //// The next four are from class 1
            //new double[] {  2,  1,  1 },
            //new double[] {  1,  1,  2 },
            //new double[] {  1,  2,  2 },
            //new double[] {  3,  1,  2 },

            //// The last three are from class 2
            //new double[] { 11,  5,  4 },
            //new double[] { 15,  5,  6 },
            //new double[] { 10,  5,  6 },
            //};

            //int[] trainOutputs =
            //{
            //    0, 0,        // First two from class 0
            //    1, 1, 1, 1,  // Next four from class 1
            //    2, 2, 2      // Last three from class 2
            //};

            //double[][] testInputs =
            //{
            //// The first two are from class 0
            //new double[] { -3, -1, -1 },
            //new double[] { -9, -7, -5 },
            //};

            //int[] testOutputs =
            //{
            //    0, 0,        // First two from class 0
            //};

            DataTable dataTrain = new CsvReader(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\CSV\train\train.csv", true).ToTable();
            DataTable dataTest = new CsvReader(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\CSV\test\testWithLabels.csv", true).ToTable();

            // I/O data //
            int[] trainOutputs = dataTrain.Columns["label"].ToArray<int>();
            dataTrain.Columns.Remove("label");
            double[][] trainInputs = dataTrain.ToJagged<double>();

            int[] testOutputs = dataTest.Columns["label"].ToArray<int>();
            dataTest.Columns.Remove("label");
            double[][] testInputs = dataTest.ToJagged<double>();
            // I/O data //

            // knn //
            var knn = new KNN(trainInputs, trainOutputs, 4);
            var machineKNN = knn.MachineLearning();
            int[] predictedKNN = machineKNN.Decide(testInputs);
            machineKNN.Save(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\models\knn.bin");

            OutputResultsСlassifier showKNN = new OutputResultsСlassifier(machineKNN, testInputs, testOutputs);
            showKNN.SavePredicted(predictedKNN);
            showKNN.SaveAccuracy();
            // knn //

            // nb //
            var nb = new NB(trainInputs, trainOutputs);
            var machineNB = nb.MachineLearning();
            int[] predictedNB = machineNB.Decide(testInputs);
            machineNB.Save(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\models\nb.bin");

            OutputResultsСlassifier show_nb = new OutputResultsСlassifier(machineNB, testInputs, testOutputs);
            show_nb.SavePredicted(predictedNB);
            show_nb.SaveAccuracy();
            // nb //

            // svm //
            var svm = new SVM(trainInputs, trainOutputs);
            var machineSVM = svm.MachineLearning();
            int[] predictedSVM = machineKNN.Decide(testInputs);
            machineKNN.Save(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\models\svm.bin");

            OutputResultsСlassifier showSVM = new OutputResultsСlassifier(machineKNN, testInputs, testOutputs);
            showSVM.SavePredicted(predictedSVM);
            showSVM.SaveAccuracy();
            // svm //

            // mlr //
            var mlr = new MLR(trainInputs, trainOutputs);
            var machineMLR = mlr.MachineLearning();
            int[] predictedMLR = machineKNN.Decide(testInputs);
            machineKNN.Save(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\models\mlr.bin");

            OutputResultsСlassifier showMLR = new OutputResultsСlassifier(machineMLR, testInputs, testOutputs);
            showMLR.SavePredicted(predictedSVM);
            showMLR.SaveAccuracy();
            showMLR.SaveProbabilities(machineMLR);
            // mlr //

        }
    }
}
