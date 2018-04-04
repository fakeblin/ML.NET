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
            DataTable data_train = new CsvReader(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\train.csv", true).ToTable();
            DataTable data_test = new CsvReader(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\train_lite.csv", true).ToTable();

            int[] trainOutputs = data_train.Columns["label"].ToArray<int>();
            data_train.Columns.Remove("label");
            double[][] trainInputs = data_train.ToJagged<double>();

            int[] testOutputs = data_test.Columns["label"].ToArray<int>();
            data_test.Columns.Remove("label");
            double[][] testInputs = data_test.ToJagged<double>();



            var knn = new KNN(trainInputs, trainOutputs, 4);
            var machine_knn = knn.MachineLearning();
            int [] predicted_knn = machine_knn.Decide(testInputs);

            PrintResultsСlassifier show_knn = new PrintResultsСlassifier(machine_knn, testInputs, testOutputs);
            show_knn.PrintPredicted(predicted_knn);
            show_knn.PrintAccuracy();
            show_knn.PrintProbabilities();

            Console.WriteLine("BEGIN NaiveBayes");

            var nb = new NB(trainInputs, trainOutputs);
            var machine_nb = nb.MachineLearning();
            int[] predicted_nb = machine_nb.Decide(testInputs);

            PrintResultsСlassifier show_nb = new PrintResultsСlassifier(machine_nb, testInputs, testOutputs);
            show_nb.PrintPredicted(predicted_knn);
            show_nb.PrintAccuracy();
            show_nb.PrintProbabilities();

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
