using Accord.Math.Optimization;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.Classifier
{
    /// <summary>
    /// Мультиномиальная логистическая регрессия
    /// </summary>
    class MLR : IMethodLearning<MultinomialLogisticRegression>
    {
        public double[][] DataTrainInput { get; }

        public int[] DataTrainOutput { get; }

        public double[][] DataTestInput { get; }

        public int[] DataTestOutput { get; }

        public MLR(double[][] testInputs, int[] testOutputs)
        {
            DataTestInput = testInputs;
            DataTestOutput = testOutputs;
        }

        public MultinomialLogisticRegression MachineLearning()
        {
            // Используем градиентный спуск
            var mll = new MultinomialLogisticLearning<GradientDescent>();

            return mll.Learn(DataTestInput, DataTestOutput);
        }
    }
}
