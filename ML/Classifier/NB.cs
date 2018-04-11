using Accord.MachineLearning.Bayes;
using Accord.Statistics.Distributions.Univariate;
using Accord.Statistics.Distributions.Fitting;

namespace ML.Classifier
{
    /// <summary>
    /// Наивный байесовский классификатор
    /// </summary>
    class NB : IMethodLearning<NaiveBayes<NormalDistribution>>
    {
        public double[][] DataTrainInput { get; }

        public int[] DataTrainOutput { get; }

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


