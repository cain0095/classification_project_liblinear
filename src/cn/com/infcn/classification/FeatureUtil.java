package cn.com.infcn.classification;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.NlpAnalysis;
import org.nlpcn.commons.lang.util.CollectionUtil;
import org.nlpcn.commons.lang.util.IOUtil;
import org.nlpcn.commons.lang.util.MapCount;

/**
 * 进行特征选择
 * 
 * @author ansj
 *
 */
public class FeatureUtil {

	private MapCount<String> mc = new MapCount<>();

	private MapCount<String> x2mc = new MapCount<>();

	private boolean iskafang = false;

	private boolean isIDF = false;

	private HashMap<String, MapCount<String>> x2mat = new HashMap<>();

	public static void main(String[] args) {

	}

	private MapCount<String> _self = new MapCount<>();

	private int docFreq = 100; // 总文件频率，初始值设为100防止错误

	public void addFeature(String word, double value) {
		_self.add(word.toLowerCase()); // 手动添加特征
	}

	public void explain(String content, String target) {
		List<Term> parse = NlpAnalysis.parse(content);// NLP分词

		HashSet<String> hs = new HashSet<>();

		for (Term term : parse) {

			if (term.getNatureStr() == null
					|| term.getNatureStr().startsWith("num")) {
				continue;
			}
			hs.add(term.getName());
		}
		for (String name : hs) {
			mc.add(name);
			if (x2mat.get(target) != null) {
				x2mat.get(target).add(name, 1);
			} else {
				x2mat.put(target, new MapCount<String>());
				x2mat.get(target).add(name, 1);
			}
			x2mc.add(target, 1);
		}

		docFreq++;
	}

	public void save_IDF(String path) {
		for (Entry<String, Double> entry : _self.get().entrySet()) {
			mc.add(entry.getKey(), entry.getValue());
		}

		List<Entry<String, Double>> sortMapByValue = CollectionUtil
				.sortMapByValue(mc.get(), -1);

		StringBuilder sb = new StringBuilder();

		int index = 0;
		for (Entry<String, Double> entry : sortMapByValue) {

			if (entry.getValue() < 3) {
				continue;
			}

			if (entry.getKey().trim().length() < 2) {
				continue;
			}

			sb.append(entry.getKey() + "\t" + (++index) + "\t"
					+ Math.log(docFreq / entry.getValue()) + "\n");
		}
		isIDF = true;
		IOUtil.Writer(path, "utf-8", sb.toString());
	}

	public void save_kafang(String path) {

		HashMap<String, MapCount<String>> x2final = new HashMap<>();

		MapCount<String> tempmc = new MapCount<>();

		Double sum = 0.0;

		Double a, b, c, d;

		for (Entry<String, Double> iter : x2mc.get().entrySet()) {
			sum = sum + iter.getValue();
		}

		for (Entry<String, MapCount<String>> iter1 : x2mat.entrySet()) {
			String target = iter1.getKey();
			for (Entry<String, Double> iter2 : iter1.getValue().get()
					.entrySet()) {
				String name = iter2.getKey();
				a = iter2.getValue();
				b = x2mc.get().get(target) - a;
				c = mc.get().get(name) - a;
				d = sum - b - c + a;
				Double x2stat = Math.pow(a * d - b * c, 2) / (a + c) / (b + d);
				if (x2final.get(target) != null) {
					x2final.get(target).add(name, x2stat);
				} else {
					x2final.put(target, new MapCount<String>());
					x2final.get(target).add(name, x2stat);
				}
			}
		}

		StringBuilder sb = new StringBuilder();
		int index = 0;

		for (Entry<String, MapCount<String>> iter : x2final.entrySet()) {

			List<Entry<String, Double>> sortMapByValue = CollectionUtil
					.sortMapByValue(iter.getValue().get(), 1);

			int temp = 0;
			for (Entry<String, Double> entry : sortMapByValue) {

				if (++temp >= 5000)
					break;
				if (mc.get().get(entry.getKey()) < 3) {
					continue;
				}
				if (entry.getKey().trim().length() < 2) {
					continue;
				}

				if (!(tempmc.get().containsKey(entry.getKey()))) {
					sb.append(entry.getKey() + "\t" + (++index) + "\t"
							+ entry.getValue() + "\n");
					tempmc.add(entry.getKey());
				}
			}
		}
		iskafang = true;
		IOUtil.Writer(path, "utf-8", sb.toString());
	}

	public void computerIDF() {

		if (isIDF) {
			return;
		}

		for (Entry<String, Double> entry : _self.get().entrySet()) {
			mc.add(entry.getKey(), entry.getValue());
		}

		_self = new MapCount<>();

		for (Entry<String, Double> entry : mc.get().entrySet()) {

			if (docFreq < entry.getValue()) {
				mc.add(entry.getKey(), 1);
			} else {
				mc.add(entry.getKey(), Math.log(docFreq / entry.getValue()));
			}
		}

		isIDF = true;
	}

	public int getDocFreq() {
		return docFreq;
	}
	
	public void setDocFreq(int docFreq) {
		this.docFreq = docFreq;
	}
}
