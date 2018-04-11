using Accord.MachineLearning;


namespace ML.Classifier
{
    /// <summary>
    /// Метод к ближайших соседей
    /// </summary>
    class KNN : IMethodLearning<KNearestNeighbors>
    {
        public double[][] DataTrainInput { get; }

        public int[] DataTrainOutput { get; }

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
