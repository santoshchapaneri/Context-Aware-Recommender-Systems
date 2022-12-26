package carskit.alg.cars.adaptation.dependent.dev;

import carskit.alg.cars.adaptation.dependent.CAMF;
import carskit.data.setting.Configuration;
import carskit.data.structure.SparseMatrix;
import carskit.generic.ContextRecommender;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import happy.coding.io.Logs;
import happy.coding.math.Randoms;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;

public class CAMF_CUCI_LOG extends ContextRecommender{

    protected Table<Integer, Integer, Double> icBias;
    protected Table<Integer, Integer, Double> ucBias;

    public CAMF_CUCI_LOG(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);
        this.algoName = "CAMF_CUCI_LOG";
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
        /*double pred=globalMean + DenseMatrix.rowMult(P, u, Q, j);
        for(int cond:getConditions(c)){
            pred+=icBias.get(j,cond)+ucBias.get(u,cond);   
            //System.out.println("Ashish predict "+icBias );
            //System.out.println("Ashish predict "+ucBias );
            //Logs.debug("predict {} {} {}",  j, cond, icBias.get(j,cond) );
            //Logs.debug("predict {} {} {}",  u, cond, ucBias.get(j,cond) );
        }*/
        
        //Logistic Kernel
        /*double x=globalMean + DenseMatrix.rowMult(P, u, Q, j);        
        for(int cond:getConditions(c)){
            x+=icBias.get(j,cond)+ucBias.get(u,cond);            
        }
        double phix=1/(1+Math.exp(-x));
        double a=1;
        double b=5-1;
        double pred=a+b*phix;*/
        
        //RBF Kernel
        double rmin=1;
        double b = 5-rmin;
        double a = globalMean+rmin;
        //double x=globalMean + DenseMatrix.rowMult(P, u, Q, j);
        //double pred=globalMean + DenseMatrix.rowMult(P, u, Q, j);
        for(int cond:getConditions(c)){
            a+=icBias.get(j,cond)+ucBias.get(u,cond);            
        }
        
        double x=0;
        for (int f = 0; f < numFactors; f++) {
            double puf = P.get(u, f);
            double qjf = Q.get(j, f);
            x+=Math.pow((puf-qjf),2);
        }
        double kpq=Math.exp(-lRate*x);
        double pred=a+b*kpq;
        
        
        
        
        
        //System.out.println("\npredict = "+pred );
        return pred;
    }

    @Override
    protected void buildModel() throws Exception {
        //Original Code
        /*for (int iter = 1; iter <= numIters; iter++) {

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

                // update factors
                double Buc_sum=0;
                double Bic_sum=0;
                for(int cond:getConditions(ctx)){
                    double Buc=ucBias.get(u,cond);
                    double Bic=icBias.get(j,cond);
                    Buc_sum+=Math.pow(Buc, 2);
                    Bic_sum+=Math.pow(Bic, 2);
                    double sgdu = euj - regC*Buc;
                    double sgdj = euj - regC*Bic;
                    ucBias.put(u,cond, Buc+lRate*sgdu);
                    icBias.put(j,cond, Bic+lRate*sgdj);
                }

                loss += regC * Bic_sum + regC * Buc_sum;

                for (int f = 0; f < numFactors; f++) {
                    double puf = P.get(u, f);
                    double qjf = Q.get(j, f);

                    double delta_u = euj * qjf - regU * puf;
                    double delta_j = euj * puf - regI * qjf;

                    P.add(u, f, lRate * delta_u);
                    Q.add(j, f, lRate * delta_j);

                    loss += regU * puf * puf + regI * qjf * qjf;
                }

            }
        */
        
        //Logistic Kernel
       /* for (int iter = 1; iter <= numIters; iter++) {

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

                // update factors
                double Buc_sum=0;
                double Bic_sum=0;
                for(int cond:getConditions(ctx)){
                    double Buc=ucBias.get(u,cond);
                    double Bic=icBias.get(j,cond);
                    Buc_sum+=Math.pow(Buc, 2);
                    Bic_sum+=Math.pow(Bic, 2);
                    double x=globalMean + DenseMatrix.rowMult(P, u, Q, j);        
                    for(int cond1:getConditions(ctx)){
                        x+=icBias.get(j,cond1)+ucBias.get(u,cond1);            
                    }
                    double phix=1/(1+Math.exp(-x));
                    
                    double sgdu = euj * Math.pow(phix,2) * Math.exp(-x) - regC*Buc;
                    double sgdj = euj * Math.pow(phix,2) * Math.exp(-x) - regC*Bic;
                    ucBias.put(u,cond, Buc+lRate*sgdu);
                    icBias.put(j,cond, Bic+lRate*sgdj);
                }

                loss += regC * Bic_sum + regC * Buc_sum;

                for (int f = 0; f < numFactors; f++) {
                    double puf = P.get(u, f);
                    double qjf = Q.get(j, f);
                    double x=globalMean + DenseMatrix.rowMult(P, u, Q, j);        
                    for(int cond1:getConditions(ctx)){
                        x+=icBias.get(j,cond1)+ucBias.get(u,cond1);            
                    }
                    double phix=1/(1+Math.exp(-x));
                    
                    double delta_u = euj * qjf * Math.pow(phix,2)*Math.exp(-x) - regU * puf;
                    double delta_j = euj * puf * Math.pow(phix,2)*Math.exp(-x) - regI * qjf;

                    P.add(u, f, lRate * delta_u);
                    Q.add(j, f, lRate * delta_j);

                    loss += regU * puf * puf + regI * qjf * qjf;
                }

            }*/
        
        //RBF Kernel
        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;
            for (MatrixEntry me : trainMatrix) {
                System.out.println("\nRow Entry = "+me.row() );
                System.out.println("\nCol Entry = "+me.column() );
                int ui = me.row(); // user-item
                int u= rateDao.getUserIdFromUI(ui);
                int j= rateDao.getItemIdFromUI(ui);
                int ctx = me.column(); // context
                double rujc = me.get();

                double pred = predict(u, j, ctx, false);
                double euj = rujc - pred;

                loss += euj * euj;

                // update factors
                double b=5-1;  //rmax-rmin
                double Buc_sum=0;
                double Bic_sum=0;
                for(int cond:getConditions(ctx)){
                    double Buc=ucBias.get(u,cond);
                    double Bic=icBias.get(j,cond);
                    Buc_sum+=Math.pow(Buc, 2);
                    Bic_sum+=Math.pow(Bic, 2);
                    
                    double sgdu = euj * 2*b - regC*Buc;
                    double sgdj = euj * 2*b - regC*Bic;
                    ucBias.put(u,cond, Buc+lRate*sgdu);
                    icBias.put(j,cond, Bic+lRate*sgdj);
                }

                loss += regC * Bic_sum + regC * Buc_sum;

                for (int f = 0; f < numFactors; f++) {
                    double puf = P.get(u, f);
                    double qjf = Q.get(j, f);
                    
                    double delta_u = euj * 2*b*(2*lRate*Math.exp(-lRate*Math.pow(puf-qjf,2))*(qjf-puf)) - regU * puf;
                    double delta_j = euj * 2*b*(2*lRate*Math.exp(-lRate*Math.pow(puf-qjf,2))*(puf-qjf)) - regI * qjf;

                    P.add(u, f, lRate * delta_u);
                    Q.add(j, f, lRate * delta_j);

                    loss += regU * puf * puf + regI * qjf * qjf;
                }

            }
            loss *= 0.5;

            if (isConverged(iter))
                break;
            for (MatrixEntry me : trainMatrix) {
                int u1=rateDao.getUserIdFromUI(me.row());
                int q1=rateDao.getItemIdFromUI(me.row());
                System.out.println("\n1Row Entry,context,rating,uid,iid = "+me.row()+"  "+me.column()+"  "+me.get()+"  "+u1+"  "+q1);
                for(int cond:getConditions(me.column())){
                    System.out.println("\nCondition,ubais,ibais,context = "+cond +" "+ ucBias.get(u1,cond)+" "+ icBias.get(q1,cond)+" "+me.column() );
                    
                }
                //int ui = me.row(); // user-item
                //int u= rateDao.getUserIdFromUI(ui);
                //int j= rateDao.getItemIdFromUI(ui);
                //int ctx = me.column(); // context
                //double rujc = me.get();
            }
        }// end of training

    }
}
