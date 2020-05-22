#ifndef VIGRA_PEGASOS_HXX
#define VIGRA_PEGASOS_HXX

#include <math.h>
#include <algorithm>
#include <limits>
#include "linear_algebra.hxx"
#include "multi_array.hxx"
#include "random.hxx"

namespace pegasos {

/** \addtogroup MachineLearning Machine Learning
**/
//@{

/** Linear support vector machine using the pegasos algorithm.
 *
 * Implements the pegasos stochastic sub-gradient descent 
 * algorithm in order to solve Support Vector Machines  
 * for two-class and multi-class problems.
 *
*/
class Pegasos
{
	private:
	
	std::vector<Matrix<double> > Ws;
	std::vector<int> classes;
	double lambda;
	double tolerance;
	int maxiter;	

	/* Returns true if MultiArray contains NaNs */
    template<unsigned int N, class T, class C>
    bool contains_nan(MultiArrayView<N, T, C> const & in)
    {
        for(int ii = 0; ii < in.size(); ++ii)
            if(in[ii] != in[ii])
                return true;
        return false; 
    }
    
    /* Returns true if MultiArray contains Infs */
    template<unsigned int N, class T, class C>
    bool contains_inf(MultiArrayView<N, T, C> const & in)
    {
         if(!std::numeric_limits<T>::has_infinity)
             return false;
         for(int ii = 0; ii < in.size(); ++ii)
            if(in[ii] == std::numeric_limits<T>::infinity())
                return true;
         return false;
    }	
		
	public:
	
	/** \brief default constructor */
    Pegasos()
    {
		lambda = .001;
		tolerance = .0001;
		maxiter = 20000;
    }
	
	/** \brief default constructor
	 *
	 * \param inLambda	Regularization parameter   
	 * \param inTolerance	Threshold tolerance	
	 * \param inMaxiter	Max iterations for gradient loop
	 *
	*/	
    Pegasos(double inLambda, double inTolerance, int inMaxiter)
    {
		lambda = inLambda;
		tolerance = inTolerance;
		maxiter = inMaxiter;
    }
    
	/** \brief learn from two-class or multi-class data
	 *
	 * \param features	s * f feature matrix where s is the number of samples and f is the number of features  
	 * \param labels	s * 1 labels matrix where s is the number of samples
	 * \param random	random number generator
	 *
	*/	
    template <class T, class RandomGenerator>
    void train( MultiArrayView<2, T>   const & features,
                MultiArrayView<2, int> const & labels,
                RandomGenerator        const & random)
    {
		Matrix<int>    svmReadyLabels = labels;
		int sti, rni, rwi, cli, csi;
		double stpsiz;
		int samplesiz = features.shape(0);
		int vecsiz = features.shape(1);
		int y_i;
		Matrix<double> X_i;
		Matrix<double> W(Shape2(vecsiz, 1), 0.0);
		Matrix<double> W_prev = W;
		Matrix<double> W_t;
		double dotwx;
		double dist;
		
		int labsiz = 0;
		int targetClass;
		int clasize;

		Ws.clear();
		classes.clear();

		/* Test for NaNs */
		if (contains_nan(features)) {
			std::cerr<<"\nLearning Error: features matrix contains NaNs.\n";
			return;
		}		
		
		/* Test for Infs */
		if (contains_inf(features)) {
			std::cerr<<"\nLearning Error: features matrix contains Infs.\n";
			return;
		}
	
		/* Test equal sample size for features and labels */
		if (features.shape(0) != labels.shape(0)) {
			std::cerr<<"\nLearning Error: features and labels sample number mismatch.\n";
			return;
		}
		
		/* Calculate number of classes */
		targetClass = 0;
		for (rwi=0; rwi<samplesiz; rwi++) {
			targetClass = labels(rwi,0);
			if (std::find(classes.begin(), classes.end(), targetClass) == classes.end())
				classes.push_back(targetClass);
		}	
		
		/* Check for multi-class problem */
		if (classes.size() == 2 && find(classes.begin(), classes.end(), 1) != classes.end() && find(classes.begin(), classes.end(), -1) != classes.end())
			clasize = 1;
		else
			clasize = classes.size();
		
		/* Loop for each class */
		for (csi=0; csi<clasize; csi++) {
		
			/* Prepare multi-class labels for SVM (0, -1) */
			if (clasize != 1) {
				targetClass = classes[csi];
				for (rwi=0; rwi<samplesiz; rwi++) {
					if (labels(rwi,0) == targetClass)
						svmReadyLabels(rwi,0) = 1;
					else
						svmReadyLabels(rwi,0) = -1;
				}
			}
			
			/* Pegasos gradient descent*/
			for (sti=0; sti<maxiter; sti++) {
				rni = random.uniformInt(samplesiz);
				X_i = (features.subarray(Shape2(rni,0), Shape2(rni+1,vecsiz))).transpose();
				y_i = svmReadyLabels(rni,0); 

				stpsiz = 1/(lambda*(sti+1));

				dotwx = dot(W,X_i);		
				
				if ((double)y_i*dotwx < 1.0) 
					W = (1-stpsiz*lambda)*W + stpsiz*(double)y_i*X_i;	
				else if ((double)y_i*dotwx >= 1.0) 
					W = (1-stpsiz*lambda)*W;
					
				W = min(1.0,(1.0/sqrt(lambda))*norm(W))*W;
			
				dist = norm(W-W_prev);
				
				W_prev = W;					
			}

			W = W/norm(W);
			
			Ws.push_back(W);	
		}
    }
	   
	/** \brief predict labels for input features
	 *
	 * \param features	s * f feature matrix where s is the number of samples and f is the number of features  
	 * \param labels	s * 1 labels matrix where s is the number of samples
	 *
	*/	
    template <class T>
    void predict(MultiArrayView<2, T>  const & features,
                       MultiArrayView<2, int>        labels) const
    {
		int rwi, cli, csi;
		int samplesiz = features.shape(0);
		int vecsiz = features.shape(1);
		Matrix<double> X_i;
		double y_i;
		double y_i_max = 0.0;
		Matrix<double> W(Shape2(vecsiz, 1), 0.0);
		
		/* Two-class with labels (1, -1) */
		if (Ws.size() == 1) { 
			for (rwi=0; rwi<samplesiz; rwi++) {
				X_i = (features.subarray(Shape2(rwi,0), Shape2(rwi+1,vecsiz))).transpose();
				y_i =  dot(Ws[0], X_i);

				if (y_i  < 0.0)
					labels(rwi,0) = -1;
				else
					labels(rwi,0) = 1;	
			}
		/* Multi-class labels */
		} else { 
			for (rwi=0; rwi<samplesiz; rwi++) {
				X_i = (features.subarray(Shape2(rwi,0), Shape2(rwi+1,vecsiz))).transpose();
				
				y_i_max = 0.0;
				for (csi=0 ; csi<classes.size(); csi++) {
					y_i = dot(Ws[csi],  X_i);
					
					if (y_i > 0 && y_i > y_i_max) {
						y_i_max = y_i;
						labels(rwi,0) = classes[csi];
					}
				}
			}
		}
    }
	
};

//@}

} // namespace pegasos

#endif // VIGRA_PEGASOS_HXX
