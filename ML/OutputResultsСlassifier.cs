using Accord.MachineLearning;
using Accord.Math.Optimization;
using Accord.Statistics.Analysis;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML
{
    class OutputResultsСlassifier
    {
        private IClassifier<double[], int> Сlassifier { get; }
        private int[] TestOutputs { get; }
        private double[][] TestInputs { get; }

        /// <summary>
        /// Если существует столбец label
        /// </summary>
        /// <param name="classifier"></param>
        /// <param name="testInputs"></param>
        /// <param name="testOutputs"></param>
        public OutputResultsСlassifier(IClassifier<double[], int> classifier, double[][] testInputs, int[] testOutputs)
        {
            Сlassifier = classifier;
            TestOutputs = testOutputs;
            TestInputs = testInputs;
        }

        public OutputResultsСlassifier(IClassifier<double[], int> classifier, double[][] testInputs)
        {
            Сlassifier = classifier;
            TestInputs = testInputs;
        }

        /// <summary>
        /// Вывод полученных и ожидаемых значений
        /// </summary>
        /// <param name="predicted">Предсказанные значения</param>
        /// <param name="testOutputs">Истинные значения</param>
        public void PrintPredicted(int[] predicted)
        {
            int i = 0;           
            if (TestOutputs.Length == 0)
            {
                Console.WriteLine("results for {0}: (counter of numbers, predict)", Сlassifier);
            }
            else
            {
                Console.WriteLine("results for {0}: (predict, real labels)", Сlassifier);
            }

            foreach (int pred in predicted)
            {
                if (TestOutputs.Length == 0)
                {
                    Console.Write("({0}, {1}) ", i, pred);
                }
                else
                {
                    Console.Write("({0},{1}) ", pred, TestOutputs[i]);
                }
                i++;
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Вывод точности в процентах.
        /// Для вычисления должен быть параметр testOutputs - ожидаемые значения
        /// </summary>
        public void PrintAccuracy()
        {
            var cm = GeneralConfusionMatrix.Estimate(Сlassifier, TestInputs, TestOutputs);
            Console.WriteLine("Accuracy for {0}: {1} %", Сlassifier, Math.Round(cm.Accuracy, 3) * 100);
        }

        /// <summary>
        /// Вывод вероятности принадлежности каждого объекта к каждому классу.
        /// Для вычисления должен быть параметр testOutputs - ожидаемые значения
        /// </summary>
        public void PrintProbabilities(MultinomialLogisticRegression mlr)
        {
            double[][] probabilities = mlr.Probabilities(TestInputs);

            Console.WriteLine("Probabilities for {0}", Сlassifier);
            for (int m = 0; m < probabilities.Count(); m++)
            {
                for (int n = 0; n < probabilities[m].Count(); n++)
                {
                    Console.WriteLine("([{0}, {1}]: {2})", m, n, probabilities[m][n]);
                }
            }
        }

        private string InitialTime()
        {
            string timeBefore = DateTime.Now.ToString();
            string timeAfter = "";

            foreach (char c in timeBefore)
            {
                if (c == ' ')
                    timeAfter += '-';
                else if (c == ':')
                    timeAfter += '_';
                else
                    timeAfter += c;
            }
            return timeAfter;
        }

        public void SaveProbabilities(MultinomialLogisticRegression mlr, string path = @"H:\Documents\Visual Studio 2015\Projects\ML\ML\SaveResults\")
        {
            string timeAfter = InitialTime();
            double[][] probabilities = mlr.Probabilities(TestInputs);

            for (int m = 0; m < probabilities.Count(); m++)
            {
                for (int n = 0; n < probabilities[m].Count(); n++)
                {
                    using (FileStream fs = new FileStream(path + timeAfter + "_Probabilities" + Сlassifier + ".txt", FileMode.Append))
                    {
                        using (StreamWriter writer = new StreamWriter(fs))
                        {
                            writer.WriteLine("([{0}, {1}]: {2})", m, n, probabilities[m][n]);
                        }
                    }
                }
            }
        }

        public void SaveAccuracy(string path = @"H:\Documents\Visual Studio 2015\Projects\ML\ML\SaveResults\")
        {
            string timeAfter = InitialTime();

            var cm = GeneralConfusionMatrix.Estimate(Сlassifier, TestInputs, TestOutputs);

            using (FileStream fs = new FileStream(path + timeAfter + "_Accuracy" + Сlassifier + ".txt", FileMode.CreateNew))
            {
                using (StreamWriter writer = new StreamWriter(fs))
                {
                    writer.WriteLine("Accuracy for {0}: {1} %", Сlassifier, Math.Round(cm.Accuracy, 3) * 100);
                }
            }
        }

        public void SavePredicted(int[] predicted, string path = @"H:\Documents\Visual Studio 2015\Projects\ML\ML\SaveResults\")
        {
            string timeAfter = InitialTime();
            int i = 0;

            foreach (int pred in predicted)
            {
                if (TestOutputs.Length == 0)
                {
                    using (FileStream fs = new FileStream(path + timeAfter + "_Predicted" + Сlassifier + ".txt", FileMode.Append))
                    {
                        using (StreamWriter writer = new StreamWriter(fs))
                        {
                            if (i == 0)
                            {
                                writer.WriteLine("results: (counter of numbers, predict)");
                            }
                            else if (i != 0 && i % 20 == 0)
                            {
                                writer.WriteLine("({0},{1}) ", i, pred);
                            }
                            else
                            {
                                writer.Write("({0},{1}) ", i, pred);
                            }
                        }
                    }
                }
                else
                {
                    using (FileStream fs = new FileStream(path + timeAfter + "_Predicted" + Сlassifier + ".txt", FileMode.Append))
                    {
                        using (StreamWriter writer = new StreamWriter(fs))
                        {
                            if (i == 0)
                            {
                                writer.WriteLine("results: (predict, real labels)");
                            }
                            else if (i != 0 && i % 20 == 0)
                            {
                                writer.WriteLine("({0},{1}) ", pred, TestOutputs[i]);
                            }
                            else
                            {
                                writer.Write("({0},{1}) ", i, pred);
                            }
                        }
                    }
                    i++;
                }
            }
        }
    }
}
