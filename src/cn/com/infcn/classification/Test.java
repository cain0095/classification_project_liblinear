package cn.com.infcn.classification;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.List;
import java.util.Random;

import javax.print.DocFlavor.STRING;

import org.ansj.app.keyword.KeyWordComputer;
import org.ansj.app.keyword.Keyword;
import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.NlpAnalysis;
import org.nlpcn.commons.lang.util.IOUtil;
import org.nlpcn.commons.lang.util.StringUtil;
import org.nlpcn.commons.lang.util.MapCount;

import de.bwaldvogel.liblinear.SolverType;

public class Test {
	public static void main(String[] args) throws Exception {
//		 feature() ;

//		 Classification cu = new Classification("feature.dic", "model.bin",
//		 "target.map");
//		
//		 cu.crossValidation(1.0, 0.01, SolverType.L2R_LR);
//
//		 String content =
//		 "闸门与启闭机安装是水电站发电，防洪，渡汛的关键项目，要与土建紧密配合，二者的施工应互相穿插，精心组织，施工中既要不相互干扰，又要齐头并进，单一考虑各自为政的施工计划将会影响双方的施工进度而最终影响整体工程进程。";		 		
//		 System.out.println(cu.predict(content)); ;
//
		 long endtime;
		 long starttime= System.currentTimeMillis();
		 Classification cu = new Classification("feature_kafang.dic", "model.bin",
		 "target.map");
		
		 BufferedReader br = IOUtil.getReader("fl.txt", "utf-8");
		
		 String temp = null;
		
		 int fcount =0;
		
		 while ((temp = br.readLine()) != null) {
		 System.out.println(++fcount);
		 String[] split = temp.split("\t");
		 
		 
		 String content = split[1];
		 String target = split[0];
		 
		 cu.add(target, content);

		 }

		 cu.printTargetmc(); //输出样本的类别分布至target2.map
		
//		 cu.getOptparam(0.6, 1, 0.05, 0.01, 0.1, 2, SolverType.L2R_LR);
		
			 
//		 System.out.println("***********************\nStarting crossvalidation.\n***********************");
//		 cu.crossValidation(0.8, 0.01, SolverType.L2R_LR);
//		 endtime=System.currentTimeMillis();
//		 System.out.println("本次交叉验证时间为："+(endtime-starttime)/1000+"s");
		         
		 starttime= System.currentTimeMillis();
		 cu.saveModel(0.8, 0.01, SolverType.L2R_LR);
		 endtime=System.currentTimeMillis();
		 System.out.println("本次运行时间为："+(endtime-starttime)/1000+"s");
		 
		 cu.batchpredict("fl.txt", "fl_result.txt");
//		 	
	}

	private static void feature() throws FileNotFoundException,
			UnsupportedEncodingException, IOException {
		BufferedReader br = IOUtil.getReader("fl.txt", "utf-8");

		String temp = null;

		FeatureUtil fu = new FeatureUtil();

		int ok = 0;

		while ((temp = br.readLine()) != null) {
			String[] split = temp.split("\t");

			fu.explain(split[1],split[0]);

			if (ok++ % 1000 == 0) {
				System.out.println(ok);
			}
		}

		fu.save_IDF("feature_IDF.dic");
		fu.save_kafang("feature_kafang.dic");
	}    
}  	  
	