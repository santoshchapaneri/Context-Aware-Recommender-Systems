package carskit.alg.cars.adaptation.dependent.dev;

import carskit.alg.cars.adaptation.dependent.CAMF;
import carskit.data.setting.Configuration;
import carskit.data.structure.SparseMatrix;
import carskit.generic.ContextRecommender;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import happy.coding.io.Logs;
import happy.coding.math.Randoms;
import java.util.List;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;

public class CAMF_CUCI_RBFK extends ContextRecommender{

    protected Table<Integer, Integer, Double> icBias;
    protected Table<Integer, Integer, Double> ucBias;

    public CAMF_CUCI_RBFK(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);
        this.algoName = "CAMF_CUCI_RBFK";
    }

    protected void initModel() throws Exception {

        super.initModel();


        ucBias= HashBasedTable.create();
        icBias= HashBasedTable.create();

        for(int u=0;u<numUsers;++u)
            for(int c=0;c<numConditions;++c)
                ucBias.put(u,c, Randoms.gaussian(initMean,initStd));

        for(int i=0;i<numItems;++i)
            for(int c=0;c<numConditions;++c)
                icBias.put(i,c, Randoms.gaussian(initMean,initStd));

    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        //Original Code
        double pred=globalMean + DenseMatrix.rowMult(P, u, Q, j);
        for(int cond:getConditions(c)){
            pred+=icBias.get(j,cond)+ucBias.get(u,cond);   
            //System.out.println("Ashish predict "+icBias );
            //System.out.println("Ashish predict "+ucBias );
            //Logs.debug("predict {} {} {}",  j, cond, icBias.get(j,cond) );
            //Logs.debug("predict {} {} {}",  u, cond, ucBias.get(j,cond) );
        }
        
        //System.out.println("\npredict = "+pred );
        return pred;
    }

    @Override
    protected void buildModel() throws Exception {
        //Original Code
        double gamma = 0.01;
        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;
            for (MatrixEntry me : trainMatrix) {

                int ui = me.row(); // user-item
                int u= rateDao.getUserIdFromUI(ui);
                int j= rateDao.getItemIdFromUI(ui);
                int ctx = me.column(); // context
                double rujc = me.get();

                double pred = predict(u, j, ctx, false);
                double euj = rujc - pred;

                loss += 2 * (1 - Math.exp(-gamma*euj*euj));

                // update factors
                double Buc_sum=0;
                double Bic_sum=0;
                for(int cond:getConditions(ctx)){
                    double Buc=ucBias.get(u,cond);
                    double Bic=icBias.get(j,cond);
                    Buc_sum+=Math.pow(Buc, 2);
                    Bic_sum+=Math.pow(Bic, 2);
                    double sgdu = 4*gamma*euj*Math.exp(-gamma*euj*euj) - regC*Buc;
                    double sgdj = 4*gamma*euj*Math.exp(-gamma*euj*euj) - regC*Bic;
                    ucBias.put(u,cond, Buc+lRate*sgdu);
                    icBias.put(j,cond, Bic+lRate*sgdj);
                }

                loss += regC * Bic_sum + regC * Buc_sum;

                for (int f = 0; f < numFactors; f++) {
                    double puf = P.get(u, f);
                    double qjf = Q.get(j, f);

                    double delta_u = 4*gamma*euj*Math.exp(-gamma*euj*euj) * qjf - regU * puf;
                    double delta_j = 4*gamma*euj*Math.exp(-gamma*euj*euj) * puf - regI * qjf;

                    P.add(u, f, lRate * delta_u);
                    Q.add(j, f, lRate * delta_j);

                    loss += regU * puf * puf + regI * qjf * qjf;
                }

            }
 
            loss *= 0.5;

            if (isConverged(iter))
                break;
        }// end of training

    }
}
