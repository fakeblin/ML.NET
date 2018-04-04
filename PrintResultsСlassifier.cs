using Accord.MachineLearning;
using Accord.Math.Optimization;
using Accord.Statistics.Analysis;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML
{
    class PrintResultsСlassifier
    {
        private IClassifier<double[], int> classifier;
        private int[] testOutputs; 
        private double[][] testInputs;

        public PrintResultsСlassifier(IClassifier<double[], int> classifier, double[][] testInputs, int[] testOutputs)
        {
            this.classifier = classifier;
            this.testOutputs = testOutputs;
            this.testInputs = testInputs;
        }

        /// <summary>
        /// Вывод полученных и ожидаемых значений
        /// </summary>
        /// <param name="predicted">Предсказанные значения</param>
        /// <param name="testOutputs">Истинные значения</param>
        public void PrintPredicted(int[] predicted)
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

        /// <summary>
        /// Вывод точности
        /// </summary>
        /// <param name="predicted">Предсказанные значения</param>
        /// <param name="testOutputs">Истинные значения</param>
        public void PrintAccuracy()
        {
            var cm = GeneralConfusionMatrix.Estimate(classifier, testInputs, testOutputs);
            Console.WriteLine("Accuracy: {0}", cm.Accuracy);
        }

        /// <summary>
        /// Вывод вероятности принадлежности каждого объекта к каждому классу
        /// </summary>
        /// <param name="testInputs"></param>
        public void PrintProbabilities()
        {
            // Create a Gradient Descent algorithm to estimate the regression
            var mll = new MultinomialLogisticLearning<GradientDescent>();

            MultinomialLogisticRegression mlr = mll.Learn(testInputs, testOutputs);

            // int[] answers = mlr.Decide(testInputs);

            double[][] probabilities = mlr.Probabilities(testInputs);

            for (int m = 0; m < probabilities.Count(); m++)
            {
                for (int n = 0; n < probabilities[m].Count(); n++)
                {
                    Console.WriteLine("([{0}, {1}]: {2})", m, n, probabilities[m][n]);
                }
            }
        }
    }
}
