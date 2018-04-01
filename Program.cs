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

namespace ML
{
    class Program
    {
        static void Main(string[] args)
        {
            DataTable data_train = new CsvReader(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\train.csv", true).ToTable();
            DataTable data_test = new CsvReader(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\train_lite.csv", true).ToTable();

            // Convert the DataTable to input and output vectors (train and test)
            int[] trainOutputs = data_train.Columns["label"].ToArray<int>();
            data_train.Columns.Remove("label");
            double[][] trainInputs = data_train.ToJagged<double>();

            int[] testOutputs = data_test.Columns["label"].ToArray<int>();
            data_test.Columns.Remove("label");
            double[][] testInputs = data_test.ToJagged<double>();

            var knn = new KNearestNeighbors(k: 4);

            knn.Learn(trainInputs, trainOutputs);
            int[] predicted = knn.Decide(testInputs);

            // print result
            int i = 0;
            Console.WriteLine("results - (predict ,real labels)");
            foreach (int pred in predicted)
            {

                Console.Write("({0},{1} )", pred, testOutputs[i]);
                i++;
            }

            //calculate the accuracy
            double error = new ZeroOneLoss(testOutputs).Loss(predicted);

            Console.WriteLine("\n accuracy: {0}", 1 - error);

            // consider the decrease in the dimension of features using PCA
            var pca = new PrincipalComponentAnalysis()
            {
                Method = PrincipalComponentMethod.Center,
                Whiten = true
            };

            pca.NumberOfOutputs = 2;
            MultivariateLinearRegression transform = pca.Learn(trainInputs);
            double[][] outputPCA = pca.Transform(trainInputs);

            // print it on the scatter plot
            ScatterplotBox.Show(outputPCA, trainOutputs).Hold();

            Console.ReadLine();
        }
    }
}
