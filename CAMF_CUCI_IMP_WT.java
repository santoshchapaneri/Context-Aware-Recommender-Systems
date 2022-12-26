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

public class CAMF_CUCI_IMP_WT extends ContextRecommender{

    protected Table<Integer, Integer, Double> icBias;
    protected Table<Integer, Integer, Double> ucBias;
    
    public CAMF_CUCI_IMP_WT(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);
        this.algoName = "CAMF_CUCI_IMP_WT";
    }

    protected void initModel() throws Exception {

        super.initModel();
        userItemsCache = trainMatrix.rowColumnsCache(cacheSpec);
        itemUsersCache=trainMatrix.columnRowsCache(cacheSpec);
        //H = new DenseMatrix(numConditions, numFactors);
        //H.init(initMean, initStd);
        
        ucBias= HashBasedTable.create();
        icBias= HashBasedTable.create();
        //System.out.println("\nH = "+H);
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
        int i=0;
        for(int cond:getConditions(c)){
            pred+=icBias.get(j,cond)+ucBias.get(u,cond); 
            pred+=DenseMatrix.rowMult(P, u, H, cond)+DenseMatrix.rowMult(Q, j, H, cond);
            //System.out.println("Ashish predict "+ DenseMatrix.rowMult(P, u, H, cond));
            //System.out.println("Ashish predict1 "+DenseMatrix.rowMult(Q, j, H, cond) );
            //Logs.debug("predict {} {} {} {}", u, j, cond, i );
            i++;
            //Logs.debug("predict {} {} {}",  u, cond, ucBias.get(j,cond) );
        }
        
        //System.out.println("\npredict = "+pred );
        return pred;
    }

    @Override
    protected void buildModel() throws Exception {
        //Original Code
        
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

                loss += euj * euj;
                
                List<Integer> items = userItemsCache.get(u);
                double wu = Math.sqrt(items.size());
                
                List<Integer> users = itemUsersCache.get(j);                
                double wi = Math.sqrt(users.size());
                // update factors
                double Buc_sum=0;
                double Bic_sum=0;
                double Hc_sum=0;
                for(int cond:getConditions(ctx)){
                    double Buc=ucBias.get(u,cond);
                    double Bic=icBias.get(j,cond);
                    Buc_sum+=Math.pow(Buc, 2);
                    Bic_sum+=Math.pow(Bic, 2);
                    double sgdu = wu > 0 ? 2*euj - regC*Buc/wu : 2*euj - regC*Buc;
                    double sgdj = wi > 0 ? 2*euj - regC*Bic/wi : 2*euj - regC*Bic;
                    ucBias.put(u,cond, Buc+lRate*sgdu);
                    icBias.put(j,cond, Bic+lRate*sgdj);
                    for (int f = 0; f < numFactors; f++) {
                        double hcf = H.get(cond, f);
                        Hc_sum+=Math.pow(hcf, 2);
                        double sgdh = 2*euj*(P.get(u, f)+Q.get(j, f)) - regC*hcf;
                        H.add(cond,f, lRate*sgdh);
                    }
                }
                double Bicw=wi > 0 ? regC * Bic_sum/wi : regC * Bic_sum;
                double Bucw=wu > 0 ? regC * Buc_sum/wu : regC * Buc_sum;
                loss += Bicw + Bucw + regC * Hc_sum;

                for (int f = 0; f < numFactors; f++) {
                    double puf = P.get(u, f);
                    double qjf = Q.get(j, f);
                    double h_sum=0;
                    for(int cond:getConditions(ctx)){
                        h_sum+=H.get(cond, f);
                    }
                    double delta_u = wu > 0 ? 2*euj * (qjf+h_sum) - regU * puf/wu : 2*euj * (qjf+h_sum) - regU * puf;
                    double delta_j = wi > 0 ? 2*euj * (puf+h_sum) - regI * qjf/wi : 2*euj * (puf+h_sum) - regI * qjf;

                    P.add(u, f, lRate * delta_u);
                    Q.add(j, f, lRate * delta_j);
                    double puw= wu > 0 ? regU * puf * puf/wu : regU * puf * puf;
                    double qjw= wi > 0 ? regI * qjf * qjf/wi : regI * qjf * qjf;
                    loss += puw + qjw;
                }

            }
 
            loss *= 0.5;

            if (isConverged(iter))
                break;
        }// end of training

    }
}
