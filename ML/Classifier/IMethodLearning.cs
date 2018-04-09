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

namespace ML.Classifier
{
    interface IMethodLearning<T>
    {
        /// <summary>
        /// Обучающаяся выборка - входные параметры
        /// </summary>
        double[][] DataTrainInput { get; }

        /// <summary>
        /// Обучающаяся выборка - входные параметры
        /// </summary>
        int[] DataTrainOutput { get; }

        /// <summary>
        /// Обучение 
        /// </summary>
        /// <param name="trainInputs">входная выборка</param>
        /// <param name="trainOutputs">целевые переменные</param>
        /// <returns></returns>
        T MachineLearning();
    }
}
