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
    interface IMethodLearning
    {
        /// <summary>
        /// Обучающаяся выборка
        /// </summary>
        DataTable data_train { get; set; }
        /// <summary>
        /// Тестовая выборка
        /// </summary>
        DataTable data_test { get; set; }

        /// <summary>
        /// Конвертация значений к параметрам, с которыми можно работать
        /// </summary>
        /// <param name="data">Выборка</param>
        /// <returns></returns>
        double[][] ConvertToInput();

        /// <summary>
        /// Конвертация целевых переменных к параметрам, с которыми можно работать
        /// </summary>
        /// <param name="data">Выборка</param>
        /// <returns></returns>
        int[] ConvertToOutput();

        /// <summary>
        /// Обучение 
        /// </summary>
        /// <param name="trainInputs">входная выборка</param>
        /// <param name="trainOutputs">целевые переменные</param>
        /// <returns></returns>
        void MachineLearning(double[][] trainInputs, int[] trainOutputs);

        /// <summary>
        /// Вывод полученных и ожидаемых значений
        /// </summary>
        /// <param name="predicted">Предсказанные значения</param>
        /// <param name="testOutputs">Истинные значения</param>
        void PrintPredicted(int[] predicted, int[] testOutputs);

        /// <summary>
        /// Вывод вероятности принадлежности каждого объекта к каждому классу
        /// </summary>
        /// <param name="testInputs"></param>

        void PrintProbabilities(int[] testInputs);

        /// <summary>
        /// Вывод точности
        /// </summary>
        /// <param name="predicted">Предсказанные значения</param>
        /// <param name="testOutputs">Истинные значения</param>
        void PrintAccuracy(int[] predicted, int[] testOutputs);
    }
}
