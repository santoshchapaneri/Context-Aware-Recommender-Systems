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


public class CAMF_CUCI_IFIMP extends ContextRecommender{
    protected DenseMatrix Y;
    protected Table<Integer, Integer, Double> icBias;
    protected Table<Integer, Integer, Double> ucBias;

    public CAMF_CUCI_IFIMP(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);
        this.algoName = "CAMF_CUCI_IFIMP";
    }

    protected void initModel() throws Exception {

        super.initModel();
        
        Y = new DenseMatrix(numItems, numFactors);
        Y.init(initMean, initStd);
        userItemsCache = trainMatrix.rowColumnsCache(cacheSpec);

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
        if(isUserSplitting)
            u = userIdMapper.contains(u,c) ? userIdMapper.get(u,c) : u;
        if(isItemSplitting)
            j = itemIdMapper.contains(j,c) ? itemIdMapper.get(j,c) : j;
        return predict1(u,j,c);
    }
    
    
    protected double predict1(int u, int j, int c) throws Exception {
        double icB,ucB,PH,QH,YQ,YH;
        double pred = globalMean + DenseMatrix.rowMult(P, u, Q, j);
        for(int cond:getConditions(c)){            
            icB=icBias.get(j,cond);
            ucB=ucBias.get(u,cond);
            PH=DenseMatrix.rowMult(P, u, H, cond);
            QH=DenseMatrix.rowMult(Q, j, H, cond);
            pred+=icB+ucB+PH+QH;
            //Logs.debug("icB= {} ucB= {} PH= {} QH= {} cond= {}",  icB, ucB, PH,QH,cond );
            //pred+=icBias.get(j,cond)+ucBias.get(u,cond);
            //pred+=DenseMatrix.rowMult(P, u, H, cond)+DenseMatrix.rowMult(Q, j, H, cond);
        }
        
        List<Integer> items = userItemsCache.get(u);
        double w = Math.sqrt(items.size());
        for (int k : items){
            YQ=DenseMatrix.rowMult(Y, k, Q, j);
            Y.row(k);
            //Logs.debug("YQ= {} k= {} ",  YQ, k );
            pred += YQ / w;
            for(int cond:getConditions(c)){
                YH=DenseMatrix.rowMult(Y, k, H, cond);
                pred += YH / w;
                //Logs.debug("YH= {} k= {} cond= {} ",  YH,k,cond );
            }
        }
        //Logs.debug("pred= {} ",  pred );
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
                double w = Math.sqrt(items.size());
                
                double[] sum_ys = new double[numFactors];
                for (int f = 0; f < numFactors; f++) {
                    double sum_f = 0;
                    for (int k : items)
                        sum_f += Y.get(k, f);

                    sum_ys[f] = w > 0 ? sum_f / w : sum_f;
                }
                
                // update factors
                double Buc_sum=0;
                double Bic_sum=0;
                double Hc_sum=0;
                for(int cond:getConditions(ctx)){
                    double Buc=ucBias.get(u,cond);
                    double Bic=icBias.get(j,cond);
                    Buc_sum+=Math.pow(Buc, 2);
                    Bic_sum+=Math.pow(Bic, 2);
                    double sgdu = 2*euj - regC*Buc;
                    double sgdj = 2*euj - regC*Bic;
                    ucBias.put(u,cond, Buc+lRate*sgdu);
                    icBias.put(j,cond, Bic+lRate*sgdj);
                    for (int f = 0; f < numFactors; f++) {
                        double hcf = H.get(cond, f);
                        Hc_sum+=Math.pow(hcf, 2);
                        double sgdh = 2*euj*(P.get(u, f)+Q.get(j, f)+sum_ys[f]) - regC*hcf;
                        H.add(cond,f, lRate*sgdh);
                    }
                }

                loss += regC * Bic_sum + regC * Buc_sum + regC * Hc_sum;

                
                
                for (int f = 0; f < numFactors; f++) {
                    double puf = P.get(u, f);
                    double qjf = Q.get(j, f);
                    double h_sum=0;
                    for(int cond:getConditions(ctx)){
                        h_sum+=H.get(cond, f);
                    }
                    double delta_u = 2*euj * (qjf+h_sum) - regU * puf;
                    double delta_j = 2*euj * (puf+sum_ys[f]+h_sum) - regI * qjf;

                    P.add(u, f, lRate * delta_u);
                    Q.add(j, f, lRate * delta_j);

                    loss += regU * puf * puf + regI * qjf * qjf;
                    
                    for (int k : items) {
                        double ykf = Y.get(k, f);
                        double delta_y = 2*euj * (qjf + h_sum )/ w - regU * ykf;
                        Y.add(k, f, lRate * delta_y);

                        loss += regU * ykf * ykf;
                    }
                }

            }
        
            loss *= 0.5;

            if (isConverged(iter))
                break;
            
        }// end of training

    }
}
