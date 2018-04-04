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
    /// <summary>
    /// Метод к ближайших соседей
    /// </summary>
    class KNN : IMethodLearning<KNearestNeighbors>
    {
        /// <summary>
        /// Обучающаяся выборка - входные параметры
        /// </summary>
        public double[][] DataTrainInput { get; set; }

        /// <summary>
        /// Обучающаяся выборка - входные параметры
        /// </summary>
        public int[] DataTrainOutput { get; set; }

        /// <summary>
        /// Кол-во соседей
        /// </summary>
        private int k;

        /// <param name="dataTrainInput">Параметры обучаемой выборки</param>
        /// <param name="dataTrainOutput">Целевая переменная обучаемой выборки</param>
        /// <param name="k">Количество соседей</param>
        public KNN(double[][] dataTrainInput, int[] dataTrainOutput, int k)
        {
            DataTrainInput = dataTrainInput;
            DataTrainOutput = dataTrainOutput;
            this.k = k;
        }

        public KNearestNeighbors MachineLearning()
        {
            var knn = new KNearestNeighbors(k);           
            return knn.Learn(DataTrainInput, DataTrainOutput);
        }
    }
}
