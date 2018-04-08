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
    /// <summary>
    /// Наивный байесовский классификатор
    /// </summary>
    class NB : IMethodLearning<NaiveBayes<NormalDistribution>>
    {
        /// <summary>
        /// Обучающаяся выборка - входные параметры
        /// </summary>
        public double[][] DataTrainInput { get; set; }

        /// <summary>
        /// Обучающаяся выборка - входные параметры
        /// </summary>
        public int[] DataTrainOutput { get; set; }

        public NB(double[][] dataTrainInput, int[] dataTrainOutput)
        {
            DataTrainInput = dataTrainInput;
            DataTrainOutput = dataTrainOutput;
        }
        public NaiveBayes<NormalDistribution> MachineLearning()
        {
            var teacher = new NaiveBayesLearning<NormalDistribution>();

            // Set options for the component distributions
            teacher.Options.InnerOption = new NormalOptions
            {
                Regularization = 1e-5 // to avoid zero variances
            };

            // Learn the naive Bayes model
            NaiveBayes<NormalDistribution> bayes = teacher.Learn(DataTrainInput, DataTrainOutput);

            return bayes;
        }
    }
}


