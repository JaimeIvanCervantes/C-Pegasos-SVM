#include <limits>
#include <vigra/unittest.hxx>
#include <vigra/matrix.hxx>
#include <include/pegasos/pegasos.hpp>

#include "testdata.hxx"

using namespace vigra;
using namespace std;

struct PegasosTest
{
	/* Print matrix content */
    void print(MultiArray<2, double> const &matrix)
    {
		int rwi, cli;
		
		for (rwi=0; rwi<matrix.shape(0); rwi++) {
			for (cli=0; cli<matrix.shape(1); cli++) {
				std::cout<<matrix(rwi, cli)<<", ";
			}
			std::cout<<"\n";
		}
    }

	/* Tests whether pegasos rejects Matrices containing NaNs */ 
    void test_pegasos_NaNCheck()
    {
		Matrix<double> features(Shape2(150, 5), iris_dataset_features);
		Matrix<int>    labels(Shape2(150, 1), iris_dataset_labels);	

		double lambda = .0001495;
		int maxiter = 1000;
		double tolerance = .0001;
		RandomMT19937 random(1); 
	
		features(0,0) = std::numeric_limits<double>::quiet_NaN();;
	
		Pegasos	peg(lambda, tolerance, maxiter);
		peg.learn(features, labels, random);
	}		
	
	/* Tests whether pegasos rejects Matrices containing Inf */ 
    void test_pegasos_InfCheck()
    {
		Matrix<double> features(Shape2(150, 5), iris_dataset_features);
		Matrix<int>    labels(Shape2(150, 1), iris_dataset_labels);	

		double lambda = .0001495;
		int maxiter = 1000;
		double tolerance = .0001;
		RandomMT19937 random(1); 
	
		features(0,0) = std::numeric_limits<double>::infinity();
	
		Pegasos	peg(lambda, tolerance, maxiter);
		peg.learn(features, labels, random);
	}	

	/* Test that the sample size of features and labels is equal */
	void test_pegasos_sample_size() {	
		Matrix<double> features(Shape2(150, 5), iris_dataset_features);
		Matrix<int>    labels(Shape2(150, 1), iris_dataset_labels);	

		double lambda = .0001495;
		int maxiter = 1000;
		double tolerance = .0001;
		RandomMT19937 random(1); 
		
		Pegasos	peg(lambda, tolerance, maxiter);
		peg.learn(features.subarray(Shape2(1,1),Shape2(features.shape(0),features.shape(1))), labels, random);		
	}
	
	/* Deterministic test for implementation correctness with an expected error of ~4.6% */
	void test_pegasos_deterministic() {
		int rwi, cli, tsi, pri;
        Matrix<double> features(Shape2(150, 5), iris_dataset_features);
        Matrix<int>    labels(Shape2(150, 1), iris_dataset_labels);
		Matrix<int>    predictedLabels(Shape2(1, 1), 0);
		int samplesiz = features.shape(0);
		int vecsiz = features.shape(1);
		
		Matrix<double> testFeatures(Shape2(1, vecsiz), 0.0);
		Matrix<int>    testLabels(Shape2(1, 1), 0);
		Matrix<double> trainingFeatures(Shape2(samplesiz-1, vecsiz), 0.0);
		Matrix<int>    trainingLabels(Shape2(samplesiz-1, 1), 0);
		int falsePredCount;
		double trainingError;

		double lambda = .0001495;
		int maxiter = 1000;
		double tolerance = .0001;
		RandomMT19937 random(5); 
		
		prepareColumns(features, features, ZeroMean);

		falsePredCount = 0;
			
		for (tsi=0; tsi<150; tsi++) {
			Pegasos	peg(lambda, tolerance, maxiter);
			
			testFeatures = features.subarray(Shape2(tsi,0), Shape2(tsi+1,5));
			
			trainingFeatures.subarray(Shape2(0,0),Shape2(tsi,5)) = features.subarray(Shape2(0,0),Shape2(tsi,5));
			trainingFeatures.subarray(Shape2(tsi,0),Shape2(samplesiz-1,5)) = features.subarray(Shape2(tsi+1,0),Shape2(samplesiz,5));
				
			trainingLabels.subarray(Shape2(0,0),Shape2(tsi,1)) = labels.subarray(Shape2(0,0),Shape2(tsi,1));
			trainingLabels.subarray(Shape2(tsi,0),Shape2(samplesiz-1,1)) = labels.subarray(Shape2(tsi+1,0),Shape2(samplesiz,1));
			
			peg.learn(trainingFeatures, trainingLabels, random);
			peg.predictLabels(testFeatures, testLabels);	

			if (testLabels(0,0) != labels(tsi,0))
				falsePredCount++;
		}
			
		trainingError = ((double)falsePredCount/150.0)*100.0;

		cout<<endl;			
		cout<<"Expected Training Error(%): ~4.6"<<endl;		
		cout<<"Obtained Training Error(%): "<<trainingError<<endl;
		cout<<endl;	
		
		shouldEqualTolerance(trainingError, 4.6, .2);
		
	}
	
	/* Performance test based on the leave-one-out approach */
    void test_pegasos_crossvalidation()
    {
		int rwi, cli, tsi, lmi, pri;
        Matrix<double> features(Shape2(150, 5), iris_dataset_features);
        Matrix<int>    labels(Shape2(150, 1), iris_dataset_labels);
		Matrix<int>    predictedLabels(Shape2(1, 1), 0);
		int samplesiz = features.shape(0);
		int vecsiz = features.shape(1);
		
		Matrix<double> testFeatures(Shape2(1, vecsiz), 0.0);
		Matrix<int>    testLabels(Shape2(1, 1), 0);
		Matrix<double> trainingFeatures(Shape2(samplesiz-1, vecsiz), 0.0);
		Matrix<int>    trainingLabels(Shape2(samplesiz-1, 1), 0);
		int falsePredCount;
		double trainingError;
		
		int maxiter = 1000;
		double tolerance = .0001;
		double lambda;
		double lambdaMinVal = .0001;
		double lambdaMaxVal = .01;
		int lambdaSteps = 10;

		prepareColumns(features, features, ZeroMean);
		 	
		for (lmi=0; lmi<lambdaSteps; lmi++) {
			lambda = lambdaMinVal + (double)lmi*((lambdaMaxVal-lambdaMinVal)/(double)lambdaSteps);
			RandomMT19937 random(lmi); 
			maxiter = 1000 + random.uniformInt(1000);

			falsePredCount = 0;
			
			for (tsi=0; tsi<150; tsi++) {
				Pegasos	peg(lambda, tolerance, maxiter);
			
				testFeatures = features.subarray(Shape2(tsi,0), Shape2(tsi+1,5));
			
				trainingFeatures.subarray(Shape2(0,0),Shape2(tsi,5)) = features.subarray(Shape2(0,0),Shape2(tsi,5));
				trainingFeatures.subarray(Shape2(tsi,0),Shape2(samplesiz-1,5)) = features.subarray(Shape2(tsi+1,0),Shape2(samplesiz,5));
				
				trainingLabels.subarray(Shape2(0,0),Shape2(tsi,1)) = labels.subarray(Shape2(0,0),Shape2(tsi,1));
				trainingLabels.subarray(Shape2(tsi,0),Shape2(samplesiz-1,1)) = labels.subarray(Shape2(tsi+1,0),Shape2(samplesiz,1));
			
				peg.learn(trainingFeatures, trainingLabels, random);
				peg.predictLabels(testFeatures, testLabels);	

				if (testLabels(0,0) != labels(tsi,0))
					falsePredCount++;
			}
			
			trainingError = ((double)falsePredCount/150.0)*100.0;
			
			cout<<"[";
			for (pri=0;pri<20;pri++) {
				if ((double)pri < ((double)lmi/(double)lambdaSteps)*20.0)
					cout<<"=";
				else
					cout<<"-";
			}
			cout<<"]"<<endl;
			cout<<"Iteration Number: "<<lmi<<"/"<<lambdaSteps<<endl;
			cout<<"Lambda: "<<lambda<<endl;
			cout<<"Max Iterations: "<<maxiter<<endl;
			cout<<"Tolerance: "<<tolerance<<endl;
			cout<<"Training Error(%): "<<trainingError<<endl;
			cout<<endl;
				
		}	
        
    }
};

void test_pegasos_comparison_libSVM() 
{
	/*
	 * SVM comparison goes here
	*/
}

struct PegasosTestSuite
: public test_suite
{
    PegasosTestSuite()
    : test_suite("PegasosTestSuite")
    {
		add( testCase( &PegasosTest::test_pegasos_NaNCheck));
		add( testCase( &PegasosTest::test_pegasos_InfCheck));
		add( testCase( &PegasosTest::test_pegasos_sample_size));
		add( testCase( &PegasosTest::test_pegasos_deterministic));
        add( testCase( &PegasosTest::test_pegasos_crossvalidation));
    }
};

int main(int argc, char ** argv)
{
    PegasosTestSuite test;
	
    int failed = test.run(testsToBeExecuted(argc, argv));

    std::cout << test.report() << std::endl;
    return (failed != 0);
}


