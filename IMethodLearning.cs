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
    interface IMethodLearning<T>
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
        /// Обучение 
        /// </summary>
        /// <param name="trainInputs">входная выборка</param>
        /// <param name="trainOutputs">целевые переменные</param>
        /// <returns></returns>
        T MachineLearning();
    }
}
