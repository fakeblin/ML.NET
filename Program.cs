using System;
using System.Linq;
using Accord.Statistics.Models.Regression.Linear;
using Accord.Statistics.Analysis;
using Accord.IO;
using Accord.Math;
using System.Data;
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
using System.IO;
using Accord.MachineLearning.Bayes;
using Accord.Statistics.Distributions.Univariate;
using Accord.Statistics.Distributions.Fitting;

namespace ML
{
    class Program
    {
        static void Main(string[] args)
        {
            var knn = new kNN(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\train.csv",
                @"H:\Documents\Visual Studio 2015\Projects\ML\ML\train_lite.csv", 4);

            var trainOutput = knn.ConvertToOutput(knn.DataTrain);
            var trainInput = knn.ConvertToInput(knn.DataTrain);

            var testOutput = knn.ConvertToOutput(knn.DataTest);
            var testInput = knn.ConvertToInput(knn.DataTest);

            var machine_knn = knn.MachineLearning(trainInput, trainOutput);
            int [] predicted_knn = machine_knn.Decide(testInput);

            knn.PrintPredicted(predicted_knn, testOutput);
            knn.PrintAccuracy(machine_knn, testInput, testOutput);
            knn.PrintProbabilities(testInput, testOutput);

            //double loss = new ZeroOneLoss(testOutput).Loss(predicted_knn);

            //var cv = CrossValidation.Create(
            //    k: 10,
            //    learner: machine_knn,
            //    loss: (actual, expected, p) => new ZeroOneLoss(expected).Loss(actual),
            //    fit: (teacher, x, y, w) => teacher.Learn(x, y, w),
            //    x: testInput, 
            //    y: testOutput
            //);

            Console.ReadLine();
        }
    }
}
