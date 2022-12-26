package carskit.alg.cars.adaptation.dependent.dev;

import carskit.alg.cars.adaptation.dependent.CAMF;
import carskit.data.setting.Configuration;
import carskit.data.structure.SparseMatrix;
import carskit.generic.ContextRecommender;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import happy.coding.io.Logs;
import happy.coding.math.Randoms;
import java.util.ArrayList;
import java.util.List;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;

public class CAMF_CUCI_IMP_IFUI_WT_RBFK extends ContextRecommender{
    protected DenseMatrix Y;
    protected DenseMatrix G;
    protected DenseVector Y1,G1;
    protected Table<Integer, Integer, Double> icBias;
    protected Table<Integer, Integer, Double> ucBias;
    protected Table<Integer, Integer, List<Integer>> UIcache;
    protected Table<Integer, Integer, List<Integer>> IUcache;

    public CAMF_CUCI_IMP_IFUI_WT_RBFK(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);
        this.algoName = "CAMF_CUCI_IMP_IFUI_WT_RBFK";
    }

    protected void initModel() throws Exception {

        super.initModel();
        
        Integer[] user_cnt = new Integer[numUsers];
        Integer[] item_cnt = new Integer[numItems];
        Y = new DenseMatrix(numItems, numFactors);
        Y.init(initMean, initStd);
        G = new DenseMatrix(numUsers, numFactors);
        G.init(initMean, initStd);
        //System.out.println("\nTrain Matrix = "+trainMatrix);
        //Logs.debug("numItems = {} numUsers= {}",  numItems,numUsers  );
        userItemsCache = trainMatrix.rowColumnsCache(cacheSpec);
        itemUsersCache = trainMatrix.columnRowsCache(cacheSpec);
        //System.out.println("\nuserItemsCache = "+userItemsCache.size());
        //System.out.println("\nitemUsersCache = "+itemUsersCache.size());
        ucBias= HashBasedTable.create();
        icBias= HashBasedTable.create();
        UIcache= HashBasedTable.create();
        IUcache= HashBasedTable.create();
                
        for(int u=0;u<numUsers;++u){
            //Logs.debug("Step-1 ={}",u);
            UIcache.put(u, 1, new ArrayList<Integer>());
            //user_cnt[u]=0;
            //Logs.debug("Step-2 ={}",u);
            for(int c=0;c<numConditions;++c)
                ucBias.put(u,c, Randoms.gaussian(initMean,initStd));
        }
        
        for(int i=0;i<numItems;++i){
            IUcache.put(i, 1, new ArrayList<Integer>());
            //item_cnt[i]=0;
            for(int c=0;c<numConditions;++c)
                icBias.put(i,c, Randoms.gaussian(initMean,initStd));
        }
        for (MatrixEntry me : trainMatrix) {
            int ui = me.row(); // user-item
            int u= rateDao.getUserIdFromUI(ui);
            int j= rateDao.getItemIdFromUI(ui);
            
            //user_cnt[u]++;
            //item_cnt[j]++;
            //Logs.debug("Step-3 ={}",u);
            List<Integer> test_u= new ArrayList<Integer>();
            //Logs.debug("Step-4 ={}",u);
            test_u=UIcache.get(u, 1);
            //Logs.debug("Step-5 ={}",u);
            if (!test_u.contains(j)){
                //Logs.debug("Step-6 ={}",u);
                test_u.add(j);
                //Logs.debug("Step-7 ={}",u);
                UIcache.put(u, 1, test_u);
                //Logs.debug("Step-8 ={}",u);
            }
               
            List<Integer> test_i= new ArrayList<Integer>();
            test_i=IUcache.get(j, 1);
            if (!test_i.contains(u)){
                test_i.add(u);
                IUcache.put(j, 1, test_i);
            }
            
        }
        //for(int i=0;i<numItems;++i){
            //Logs.debug("j= {}  items= {} ",i, IUcache.get(i, 1).size() /*Y.row(k).add(Y.row(k+1))*/ );
        //}
    }

    
    @Override
    protected double predict(int u, int j, int c) throws Exception {
        if(isUserSplitting){
            //Logs.debug("before u= {}",  u );
            u = userIdMapper.contains(u,c) ? userIdMapper.get(u,c) : u;
            //Logs.debug("After u= {}",  u );
        }
        if(isItemSplitting){
            //Logs.debug("Before j= {} ",  j );
            j = itemIdMapper.contains(j,c) ? itemIdMapper.get(j,c) : j;
            //Logs.debug("After j= {} ",  j );
        }
        return predict1(u,j,c);
    }
    
    
    protected double predict1(int u, int j, int c) throws Exception {
        double icB,ucB,PH,QH,YQ,YH,GH,GP,YG;
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
        
        //List<Integer> items = userItemsCache.get(u);
        List<Integer> items = UIcache.get(u, 1);
        double wu = Math.sqrt(items.size());
        Y1 = new DenseVector(numFactors);
        Y1.init();
        //Logs.debug("Items  ---------------------------------------------------------------------------" /*Y.row(k).add(Y.row(k+1))*/ );
        //Logs.debug("Y1= {} ",  Y1.add(Y.row(1)) /*Y.row(k).add(Y.row(k+1))*/ );
        //Logs.debug("items= {} ",  items /*Y.row(k).add(Y.row(k+1))*/ );
        for (int k : items){
            YQ=DenseMatrix.rowMult(Y, k, Q, j);
            Y1.add(Y.row(k));
            //Logs.debug("k= {} ",  k /*Y.row(k).add(Y.row(k+1))*/ );
           // Logs.debug("YQ= {} ",  T /*Y.row(k).add(Y.row(k+1))*/ );
            pred += YQ / wu;
            for(int cond:getConditions(c)){
                YH=DenseMatrix.rowMult(Y, k, H, cond);
                pred += YH / wu;
                //Logs.debug("YH= {} k= {} cond= {} ",  YH,k,cond );
            }
        }
        
        //List<Integer> users = itemUsersCache.get(j);
        List<Integer> users = IUcache.get(j, 1);
        double wi = Math.sqrt(users.size());
        G1 = new DenseVector(numFactors);
        G1.init();
        //T = new DenseVector(numFactors);
        //T.init(0,0);
        //Logs.debug("Users  ---------------------------------------------------------------------------" /*Y.row(k).add(Y.row(k+1))*/ );
        //Logs.debug("users.size= {} ",  users.size() /*Y.row(k).add(Y.row(k+1))*/ );
        //Logs.debug("users= {} ",  users /*Y.row(k).add(Y.row(k+1))*/ );
        for (int k1 : users){
            G1.add(G.row(k1));
            //Logs.debug("k1= {} ",  k1 /*Y.row(k).add(Y.row(k+1))*/ );
            GP=DenseMatrix.rowMult(G, k1, P, u);            
           // Logs.debug("YQ= {} ",  T /*Y.row(k).add(Y.row(k+1))*/ );
            pred += GP / wi;
            //Logs.debug("GH= {} k= {} cond= {} ",  GH,users,users.size() );
            for(int cond:getConditions(c)){
                GH=DenseMatrix.rowMult(G, k1, H, cond);
                pred += GH / wi;
                //Logs.debug("GH= {} G= {} cond= {} k= {} ",  trainMatrix.,G.numRows(),users.size(),k1 );
            }
        }
        YG=Y1.inner(G1);
        pred+=YG/(wu*wi);
        //Logs.debug("pred= {} ",  pred );
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
                //Logs.debug("ui= {} u= {} j= {} ctx= {} ",  ui,u,j,ctx );
                double pred = predict(u, j, ctx, false);
                double euj = rujc - pred;

                loss += 2 * (1 - Math.exp(-gamma*euj*euj));
                
                //List<Integer> items = userItemsCache.get(u);
                List<Integer> items = UIcache.get(u, 1);
                double wu = Math.sqrt(items.size());
                
                double[] sum_ys = new double[numFactors];
                for (int f = 0; f < numFactors; f++) {
                    double sum_f = 0;
                    for (int k : items)
                        sum_f += Y.get(k, f);

                    sum_ys[f] = wu > 0 ? sum_f / wu : sum_f;
                }
                
                //List<Integer> users = itemUsersCache.get(j);
                List<Integer> users = IUcache.get(j, 1);
                double wi = Math.sqrt(users.size());
                
                double[] sum_gs = new double[numFactors];
                for (int f = 0; f < numFactors; f++) {
                    double sum_f1 = 0;
                    for (int k : users)
                        sum_f1 += G.get(k, f);

                    sum_gs[f] = wi > 0 ? sum_f1 / wi : sum_f1;
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
                    double sgdu = wu > 0 ? 4*gamma*euj*Math.exp(-gamma*euj*euj) - regC*Buc/wu : 4*gamma*euj*Math.exp(-gamma*euj*euj) - regC*Buc;
                    double sgdj = wi > 0 ? 4*gamma*euj*Math.exp(-gamma*euj*euj) - regC*Bic/wi : 4*gamma*euj*Math.exp(-gamma*euj*euj) - regC*Bic;
                    ucBias.put(u,cond, Buc+lRate*sgdu);
                    icBias.put(j,cond, Bic+lRate*sgdj);
                    for (int f = 0; f < numFactors; f++) {
                        double hcf = H.get(cond, f);
                        Hc_sum+=Math.pow(hcf, 2);
                        double sgdh = 4*gamma*euj*Math.exp(-gamma*euj*euj)*(P.get(u, f)+Q.get(j, f)+sum_ys[f]+sum_gs[f]) - regC*hcf;
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
                    double delta_u = wu > 0 ? 4*gamma*euj*Math.exp(-gamma*euj*euj) * (qjf+sum_gs[f]+h_sum) - regU * puf/wu : 4*gamma*euj*Math.exp(-gamma*euj*euj) * (qjf+sum_gs[f]+h_sum) - regU * puf;
                    double delta_j = wi > 0 ? 4*gamma*euj*Math.exp(-gamma*euj*euj) * (puf+sum_ys[f]+h_sum) - regI * qjf/wi : 4*gamma*euj*Math.exp(-gamma*euj*euj) * (puf+sum_ys[f]+h_sum) - regI * qjf;

                    P.add(u, f, lRate * delta_u);
                    Q.add(j, f, lRate * delta_j);
                    double puw= wu > 0 ? regU * puf * puf/wu : regU * puf * puf;
                    double qjw= wi > 0 ? regI * qjf * qjf/wi : regI * qjf * qjf;
                    loss += puw + qjw;
                    
                    for (int k : items) {
                        double ykf = Y.get(k, f);
                        double delta_y = wu > 0 ? 4*gamma*euj*Math.exp(-gamma*euj*euj) * (qjf + sum_gs[f] + h_sum )/ wu - regU * ykf/wu : 4*gamma*euj*Math.exp(-gamma*euj*euj) * (qjf + sum_gs[f] + h_sum ) - regU * ykf;
                        Y.add(k, f, lRate * delta_y);
                        double ykfw = wu > 0 ? regU * ykf * ykf/wu : regU * ykf * ykf;
                        loss += ykfw;
                    }
                    
                    for (int k : users) {
                        double gkf = G.get(k, f);
                        double delta_g = wi > 0 ? 4*gamma*euj*Math.exp(-gamma*euj*euj) * (puf + sum_ys[f] + h_sum )/ wi - regI * gkf/wi : 4*gamma*euj*Math.exp(-gamma*euj*euj) * (puf + sum_ys[f] + h_sum ) - regI * gkf;
                        G.add(k, f, lRate * delta_g);
                        double gkfw = wi > 0 ? regI * gkf * gkf/wi : regI * gkf * gkf;
                        loss += gkfw;
                    }
                }

            }
        
            loss *= 0.5;

            if (isConverged(iter))
                break;
            
        }// end of training

    }
}
