package cn.com.infcn.classification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.TreeSet;
import java.lang.Math;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.NlpAnalysis;
import org.nlpcn.commons.lang.tire.GetWord;
import org.nlpcn.commons.lang.tire.domain.Forest;
import org.nlpcn.commons.lang.tire.library.Library;
import org.nlpcn.commons.lang.util.IOUtil;
import org.nlpcn.commons.lang.util.StringUtil;
import org.nlpcn.commons.lang.util.MapCount;

import com.google.common.collect.HashBiMap;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;

/**
 * 分类训练
 * 
 * @author ansj
 *
 */
public class Classification {

	private String featurePath = null;

	private String modelPath;

	public static void main(String[] args) throws Exception {
	}

	private Forest forest = null;

	private int L = 0;

	private int N = 0;

	private double targetId = 0;

	private HashBiMap<String, Double> targetMap = HashBiMap.create();

	private MapCount<String> Targetmc= new MapCount<>();
	
	private List<List<Feature>> allFeature = new ArrayList<>();

	private List<Double> allTarget = new ArrayList<>();

	private String targetPath;

	private Model model;
	
	private double accuMax;
	
	private double optC;
	
	private double optEps;

	public Classification(String featurePath, String modelPath, String targetPath) throws Exception {
		this.featurePath = featurePath;
		this.modelPath = modelPath;
		this.targetPath = targetPath;
		forest = Library.makeForest(featurePath);
	}

	public void add(String target, String content) {

		if (N == 0) {
			N = getFeatureSize();
		}

		if (StringUtil.isBlank(content)) {
			return;
		}
        
		
		Targetmc.add(target);		
		Double d = targetMap.get(target);
	
		if (d == null) {
			targetMap.put(target, targetId++);
		}

		L++;

		GetWord word = forest.getWord(content);

		List<Feature> featureList = new ArrayList<>();

		TreeMap<Integer,Double> words = new TreeMap<>();

		MapCount<Integer> mc = new MapCount<Integer>() ;
		
		while ((word.getFrontWords()) != null) {
			Integer wordId = Integer.parseInt(word.getParam(0)) ;
			words.put(wordId,Double.parseDouble(word.getParam(1)));
			mc.add(wordId);
		}

		for (Entry<Integer, Double> entry : words.entrySet()) {
			featureList.add(new FeatureNode(entry.getKey(), mc.get().get(entry.getKey())));
//1 or 0			
//			featureList.add(new FeatureNode(entry.getKey(), entry.getValue()*mc.get().get(entry.getKey())));
//TF*IDF
//			featureList.add(new FeatureNode(entry.getKey(), entry.getValue()));
//IDF	
		}

		if (featureList.size() <= 0) {
		    L--;
			return;
		}

		allTarget.add(targetMap.get(target));

		allFeature.add(featureList);

	}

	public void saveModel(Double C, Double eps, SolverType solver) throws IOException {
		Problem problem = new Problem();
		problem.l = L;
		problem.n = N;

		Feature[][] X = new Feature[allFeature.size()][0];
		for (int i = 0; i < allFeature.size(); i++) {
			X[i] = allFeature.get(i).toArray(new FeatureNode[allFeature.get(i).size()]);
		}
		problem.x = X;

		double[] Y = new double[allTarget.size()];
		for (int i = 0; i < Y.length; i++) {
			Y[i] = allTarget.get(i);
		}
		problem.y = Y;

		if (solver == null)
			solver = SolverType.L2R_LR; // -s 0
		if (C == null)
			C = 1.0; // cost of constraints violation
		if (eps == null)
			eps = 0.01; // stopping criteria

		Parameter parameter = new Parameter(solver, C, eps);

		model = Linear.train(problem, parameter);

		model.save(new File(modelPath));

		// save targetPath
		IOUtil.writeMap(targetMap, targetPath, "utf-8");
	}
	
	public void getOptparam(double Cstart, double Cend, double Cstep, double Epsstart,
			double Epsend, double Epsstep,SolverType solver)throws 
			 IOException
	{ 
		FileOutputStream fos = new FileOutputStream("paramchoice.txt");
		
		Problem problem = new Problem();
		
	    problem.l = L;
	    problem.n = N;

	    Feature[][] X = new Feature[allFeature.size()][0];
	    for (int i = 0; i < allFeature.size(); i++) {
		    X[i] = allFeature.get(i).toArray(new FeatureNode[allFeature.get(i).size()]);
	    }
	    problem.x = X;

	    double[] Y = new double[allTarget.size()];
	    for (int i = 0; i < Y.length; i++) {
		    Y[i] = allTarget.get(i);
	    }
	    problem.y = Y;

	    if (solver == null)
		    solver = SolverType.L2R_LR; // -s 0
	    
	    double tempC=Cstart;
	    double tempEps=Epsstart;
	    double tempaccu=0;
		double[] result = new double[L] ;
	    accuMax=0;
	    optC=0;
	    optEps=0;
	    Parameter parameter;
	    
	    for( ;tempC<=Cend; tempC=tempC+Cstep)
	    {   
	    	for ( tempEps=Epsstart;tempEps<=Epsend; tempEps=tempEps*Epsstep)
	    	{
	   
	    		parameter = new Parameter(solver, tempC, tempEps);	
	    		Linear.crossValidation(problem, parameter, 10, result);
	            tempaccu= getAccuarcy(Y,result);
	            System.out.println("New:C="+tempC+";  Eps="+tempEps+";  Accuarcy="+tempaccu);
	            fos.write(("New:C="+tempC+";  Eps="+tempEps+";  Accuarcy="+tempaccu+"\n").getBytes());
	    		if (tempaccu>accuMax){
	    			optC=tempC;
	    			optEps=tempEps;
	    			accuMax=tempaccu;
	    			System.out.println("Better param found.");
	    		}
	    		else{
	    			System.out.println("Param remain unchanged.");	    			
	    		}
	    		System.out.println("After:C="+optC+";  Eps="+optEps+";  Accuarcy="+accuMax);	
	    	}
	    }
	}
	
	
	public void crossValidation(Double C, Double eps, SolverType solver) throws IOException {
		Problem problem = new Problem();
		problem.l = L;
		problem.n = N;

		Feature[][] X = new Feature[allFeature.size()][0];
		for (int i = 0; i < allFeature.size(); i++) {
			X[i] = allFeature.get(i).toArray(new FeatureNode[allFeature.get(i).size()]);
		}
		problem.x = X;

		double[] Y = new double[allTarget.size()];
		for (int i = 0; i < Y.length; i++) {
			Y[i] = allTarget.get(i);
		}
		problem.y = Y;

		if (solver == null)
			solver = SolverType.L2R_LR; // -s 0
		if (C == null)
			C = 1.0; // cost of constraints violation
		if (eps == null)
			eps = 0.01; // stopping criteria

		Parameter parameter = new Parameter(solver, C, eps);
		
		double[] result = new double[L] ;
		
//		System.out.println(problem.l+"  "+problem.n+"  "+problem.x.length+"  "+problem.x[2].length+"  "+problem.x[3].length+"  "+problem.y.length);
		
		Linear.crossValidation(problem, parameter, 10, result);
		
        accuMax= getAccuarcy(Y,result);
		
//			for (int i=0; i< result.length; i++){
//				System.out.println(problem.y[i]+"  "+result[i]);
//			}
			System.out.println("**********\n  accuMax="+accuMax+"\n**********");
		
	}

	private double getAccuarcy(double[] y,double[] target){
		double sum=0;
		if (y.length != target.length) throw new IllegalArgumentException("Error in crossvalidation: the lengths of the labels do not match.");
		for (int i = 0;i < y.length;i++){
//			if ((Math.abs(y[i]-target[i]))<=0.1)
			if (y[i]==target[i])
				++sum;
		}
		return (sum/y.length);
	}
		
	private int getFeatureSize() {
		try (BufferedReader reader = IOUtil.getReader(featurePath, "utf-8")) {
			int n = 0;
			while (reader.readLine() != null) {
				n++;
			}
			return n;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return 100;

	}

	public String predict(String content) throws IOException {
		if (model == null) {
			init();
		}

		GetWord word = forest.getWord(content);

		List<Feature> featureList = new ArrayList<>();

		HashSet<Integer> words = new HashSet<>();

		while ((word.getFrontWords()) != null) {
			words.add(Integer.parseInt(word.getParam(0)));
		}

		for (Integer w1 : words) {

			featureList.add(new FeatureNode(w1, 1.0));
		}

		Collections.sort(featureList, new Comparator<Feature>() {

			@Override
			public int compare(Feature o1, Feature o2) {
				return o1.getIndex() - o2.getIndex();
			}
		});

		FeatureNode[] instance = featureList.toArray(new FeatureNode[featureList.size()]);

		double predict = Linear.predict(model, instance);

		return targetMap.inverse().get(predict);
	}
	
	public void batchpredict(String testPath, String resultPath) throws IOException
	{
		BufferedReader br = IOUtil.getReader(testPath, "utf-8");

		HashMap<String, MapCount<String>> hmc= new HashMap<String,MapCount<String>>();				
		
		FileOutputStream fos = new FileOutputStream(resultPath);

		String temp = null;
        
		double corcount = 0;
		
		int fcount = 0;

		while ((temp = br.readLine()) != null) {
			System.out.println(++fcount);
			String[] split = temp.split("\t");

			String content = split[1];
			String target = split[0];

            String predict_class=predict(content);
           
            
            if (hmc.get(target)!=null){            	    
            		hmc.get(target).add(predict_class,1);
            }
            else {
				hmc.put(target,new MapCount<String>());
            	hmc.get(target).add(predict_class,1);
            }
            
            if (predict_class.equals(target))
            {
            	fos.write(("class:"+target+"  predicted class:"+predict_class+"  content:"+split[1]+"Correct!\n").getBytes());
            	++corcount;
            	continue;
		}
            else{
            	fos.write(("class:"+target+"  predicted class:"+predict_class+"  content:"+split[1]+"Incorrect!\n").getBytes());
            }
		}
		for(String temp1:hmc.keySet())
			fos.write(("\t"+temp1).getBytes());
		
		fos.write(("\n").getBytes());
		for (String temp1:hmc.keySet())
		{
			fos.write(temp1.getBytes());
			for(String temp2: hmc.keySet())
				fos.write(("\t"+hmc.get(temp1).get().get(temp2)).getBytes());				
			fos.write(("\n").getBytes());				
		}
		
		System.out.println("Accuracy="+(corcount/fcount));
		fos.write(("General Accuracy:"+(corcount/fcount)).getBytes());
		fos.flush();
		fos.close();
		
	}

	/**
	 * 加载模型
	 * 
	 * @throws IOException
	 */
	private void init() throws IOException {
		model = Model.load(new File(modelPath));
		HashMap<String, Double> loadMap = IOUtil.loadMap(targetPath, "utf-8", String.class, Double.class);
		targetMap.putAll(loadMap);
	}
	public void printTargetmc() throws IOException
    {
    	IOUtil.writeMap(Targetmc.get(), "target2.map", "utf-8");
    }
	public String mergeclass(String c){
		switch (c.charAt(0)){
		case 'a':;
		case 'b': return "a,b";
		case 'c': return "c";
		case 'd': return "d";
		case 'f':;
		case 'g':;
		case 'h': return "f,g,h";
		case 'i': return "i";
		case 'j': return "j";
		case 'k': return "k";
		case 'o':;
		case 'p': return "o,p";
		case 'q':;
		case 's':;
		case 'x': return "q,s,x";
		case 't':;
		case 'u':;
		case 'v': return "t,u,v";
		case 'r': return "r";
		case 'e':;
		case 'n': return "Deposed.";
		default : return "Class not found.";
		
		}	
	}
}

    