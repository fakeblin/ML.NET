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
                Console.WriteLine("results: (counter of numbers, predict)");
            }
            else
            {
                Console.WriteLine("results: (predict, real labels)");
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
            Console.WriteLine("Accuracy: {0}", Math.Round(cm.Accuracy, 3)*100);
        }

        /// <summary>
        /// Вывод вероятности принадлежности каждого объекта к каждому классу.
        /// Для вычисления должен быть параметр testOutputs - ожидаемые значения
        /// </summary>
        public void PrintProbabilities(MultinomialLogisticRegression mlr)
        {
            double[][] probabilities = mlr.Probabilities(TestInputs);

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
            string time_before = DateTime.Now.ToString();
            string time_after = "";

            foreach (char c in time_before)
            {
                if (c == ' ')
                    time_after += '-';
                else if (c == ':')
                    time_after += '_';
                else
                    time_after += c;
            }
            return time_after;
        }

        public void SaveProbabilities(MultinomialLogisticRegression mlr, string path = @"H:\Documents\Visual Studio 2015\Projects\ML\ML\SaveResults\")
        {
            string time_after = InitialTime();

           // using (new FileStream(path + time_after + "_Probabilities" +".txt", FileMode.CreateNew)) { }

            double[][] probabilities = mlr.Probabilities(TestInputs);

            for (int m = 0; m < probabilities.Count(); m++)
            {
                for (int n = 0; n < probabilities[m].Count(); n++)
                {
                    using(FileStream fs = new FileStream(path + time_after + "_Probabilities" + ".txt", FileMode.CreateNew))
                    {
                        StreamWriter writer = new StreamWriter(fs);
                        writer.WriteLine("([{0}, {1}]: {2})", m, n, probabilities[m][n]);
                    }
                }
            }
        }

        public void SaveAccuracy(string path = @"H:\Documents\Visual Studio 2015\Projects\ML\ML\SaveResults\")
        {
            string time_after = InitialTime();

            var cm = GeneralConfusionMatrix.Estimate(Сlassifier, TestInputs, TestOutputs);

            using (FileStream fs = new FileStream(path + time_after + "_Accuracy" + ".txt", FileMode.CreateNew))
            {
                StreamWriter writer = new StreamWriter(fs);
                writer.WriteLine("Accuracy: {0}", Math.Round(cm.Accuracy, 3) * 100);
            }
        }

        public void SavePredicted(int[] predicted, string path = @"H:\Documents\Visual Studio 2015\Projects\ML\ML\SaveResults\")
        {
            string time_after = InitialTime();


                int i = 0;
                if (TestOutputs.Length == 0)
                {
                    using (FileStream fs = new FileStream(path + time_after + "_Predicted" + ".txt", FileMode.CreateNew))
                    {
                    StreamWriter writer = new StreamWriter(fs);
                    writer.WriteLine("results: (counter of numbers, predict)");
                    }
                }
                else
                {
                    using (FileStream fs = new FileStream(path + time_after + "_Predicted" + ".txt", FileMode.CreateNew))
                    {
                        StreamWriter writer = new StreamWriter(fs);
                        writer.WriteLine("results: (predict, real labels)");
                    }
                }

                foreach (int pred in predicted)
                {
                    if (TestOutputs.Length == 0)
                    {
                        using (FileStream fs = new FileStream(path + time_after + "_Predicted" + ".txt", FileMode.Append))
                        {
                            StreamWriter writer = new StreamWriter(fs);
                            writer.WriteLine("({0}, {1}) ", i, pred);
                        }
                    }
                    else
                    {
                        using (FileStream fs = new FileStream(path + time_after + "_Predicted" + ".txt", FileMode.Append))
                        {
                            StreamWriter writer = new StreamWriter(fs);
                            writer.WriteLine("({0},{1}) ", pred, TestOutputs[i]);
                        }
                    }
                    i++;
                }
        }
    }
}
