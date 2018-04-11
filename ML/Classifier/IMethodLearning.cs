
namespace ML.Classifier
{
    interface IMethodLearning<T>
    {
        /// <summary>
        /// Обучающающая выборка - входные параметры
        /// </summary>
        double[][] DataTrainInput { get; }

        /// <summary>
        /// Обучающающая выборка - выходные параметры
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
