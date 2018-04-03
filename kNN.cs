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
using Accord.Statistics.Distributions.Univariate;
using Accord.MachineLearning.Bayes;
using Accord.Statistics;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Math.Optimization;
using Accord.Statistics.Models.Regression;

namespace ML
{
    class kNN : IMethodLearning<KNearestNeighbors>
    {
        private DataTable dataTrain;
        private DataTable dataTest;
        public DataTable DataTrain { get { return dataTrain; } set { dataTrain = value; } }
        /// <summary>
        /// Тестовая выборка
        /// </summary>
        public DataTable DataTest { get { return dataTest; } set { dataTest = value; } }

        private int k;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="pathTrain">Путь к обучаемой выборке</param>
        /// <param name="pathTest">Путь к тестовой выборке</param>
        /// <param name="k">Количество соседей</param>
        public kNN(string pathTrain, string pathTest, int k = 1, bool HasHeadersTrain = true, bool HasHeadersTest = true)
        {
            dataTrain = new CsvReader(pathTrain, HasHeadersTrain).ToTable();
            dataTest = new CsvReader(pathTest, HasHeadersTest).ToTable();
            this.k = k;
        }

        public int[] ConvertToOutput(DataTable data)
        {
            return data.Columns["label"].ToArray<int>();
        }

        public double[][] ConvertToInput(DataTable data)
        {
            data.Columns.Remove("label");
            return data.ToJagged<double>();
        }

        public KNearestNeighbors MachineLearning(double[][] trainInputs, int[] trainOutputs)
        {
            var knn = new KNearestNeighbors(k);
            knn.Learn(trainInputs, trainOutputs);
            return knn;
        }

        public void PrintPredicted(int[] predicted, int[] testOutputs)
        {
            int i = 0;
            Console.WriteLine("results: (predict, real labels)");
            foreach (int pred in predicted)
            {

                Console.Write("({0},{1}) ", pred, testOutputs[i]);
                i++;
            }
            Console.WriteLine();
        }

        public void PrintAccuracy(KNearestNeighbors classifier, double[][] testInputs, int[] testOutputs)
        {
            // double error = new ZeroOneLoss(testOutputs).Loss(predicted);
            // Console.WriteLine("Accuracy: {0}", 1 - error);
            var cm = GeneralConfusionMatrix.Estimate(classifier, testInputs, testOutputs);
            Console.WriteLine("Accuracy: {0}", cm.Accuracy);
        }

        public void PrintProbabilities(double[][] testInputs, int[] testOutputs)
        {
            // Create a Gradient Descent algorithm to estimate the regression
            var mll = new MultinomialLogisticLearning<GradientDescent>();

            MultinomialLogisticRegression mlr = mll.Learn(testInputs, testOutputs);

            int[] answers = mlr.Decide(testInputs);

            double[][] probabilities = mlr.Probabilities(testInputs);

            for (int m = 0; m < probabilities.Count(); m++)
            {
                for (int n = 0; n < probabilities[m].Count(); n++)
                {
                    Console.WriteLine("([{0}, {1}]: {2})", m, n, probabilities[m][n]);
                }
            }
        }





        //        DataTable data_train = new CsvReader(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\train.csv", true).ToTable();
        //        DataTable data_test = new CsvReader(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\train_lite.csv", true).ToTable();

        //        // Convert the DataTable to input and output vectors (train and test)
        //        int[] trainOutputs = data_train.Columns["label"].ToArray<int>();
        //        data_train.Columns.Remove("label");
        //            double[][] trainInputs = data_train.ToJagged<double>();

        //        int[] testOutputs = data_test.Columns["label"].ToArray<int>();
        //        data_test.Columns.Remove("label");
        //            double[][] testInputs = data_test.ToJagged<double>();

        //        var knn = new KNearestNeighbors(k: 4);

        //        knn.Learn(trainInputs, trainOutputs);
        //            int[] predicted = knn.Decide(testInputs);


        //        var cm = GeneralConfusionMatrix.Estimate(knn, testInputs, testOutputs);
        //        knn.Save(Path.Combine(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\", "knn.bin"));

        //            // print result
        //            int i = 0;
        //        Console.WriteLine("results - (predict ,real labels)");
        //            foreach (int pred in predicted)
        //            {

        //                Console.Write("({0},{1})", pred, testOutputs[i]);
        //                i++;
        //            }
        //    Console.WriteLine();

        //            //calculate the accuracy
        //            double error = new ZeroOneLoss(testOutputs).Loss(predicted);

        //    Console.WriteLine("\n Accuracy: {0} (cm.Accuracy:{1}) (cm.Error:{2}) (cm.Kappa:{3})", 
        //                1 - error, cm.Accuracy, cm.Error, cm.Kappa);

        //            // Create a Conjugate Gradient GradientDescent algorithm to estimate the regression
        //            var mgd = new MultinomialLogisticLearning<GradientDescent>();

        //    MultinomialLogisticRegression mlr = mgd.Learn(testInputs, testOutputs);

        //    int[] answers = mlr.Decide(testInputs);

        //    // print result
        //    int l = 0;
        //    Console.WriteLine("results - (predict ,real labels)");
        //            foreach (int ans in answers)
        //            {

        //                Console.Write("({0},{1})", ans, testOutputs[l]);
        //                l++;
        //            }

        //// And also the probability of each of the answers
        //double[][] probabilities = mlr.Probabilities(testInputs);

        //            for (int m = 0; m<probabilities.Count(); m++)
        //            {
        //                for (int n = 0; n<probabilities[m].Count(); n++)
        //                {
        //                    Console.WriteLine("([{0}, {1}]: {2})", m, n, probabilities[m][n]);
        //                }
        //            }

        //            // Now we can check how good our model is at predicting
        //            double error_reg = new ZeroOneLoss(testOutputs).Loss(answers);
        //Console.WriteLine("\n accuracy_answer: {0}", 1 - error_reg);

        //            Console.ReadLine();
    }
}
