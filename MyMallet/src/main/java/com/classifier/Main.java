package com.classifier;
/**
 * created by Shabir Ahmad
 */
import cc.mallet.classify.Classifier;
import cc.mallet.classify.Trial;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Labeling;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;

/**
 *
 */

public class Main {

    public static void main(String args[]) throws IOException {

        Main mainClass = new Main();
        MyMalletClassifier myMalletClassifier = new MyMalletClassifier();

        InstanceList trainingInstanceList = new InstanceList(myMalletClassifier.getInstancePipe());

        File file = new File("resource/data.txt");
        trainingInstanceList.addThruPipe(new CsvIterator(new FileReader(file),
                "(\\w+)\\s+(\\w+)\\s+(.*)",
                3, 2, 1));

        File file2 = new File("resource/test.txt");
        mainClass.printLabelings(myMalletClassifier.trainClassifier(trainingInstanceList), file2);

        mainClass.evaluate(myMalletClassifier.trainClassifier(trainingInstanceList), file2);

        InstanceList trinList = new InstanceList(myMalletClassifier.getInstancePipe());
        InstanceList testList = new InstanceList(myMalletClassifier.getInstancePipe());

        trinList.addThruPipe(new CsvIterator(new  FileReader("resource/traintag.txt"),"(.*)\\s+(\\w)",2,1,-1));
        testList.addThruPipe(new CsvIterator(new  FileReader("resource/testtag.txt"),"(.*)\\s+(\\w)",2,1,-1));
        Tagging tagging = new Tagging();
        tagging.run(trinList,testList);
    }

    /**
     * this example shows how to use a trained classifier to guess the class of new data.
     * @param classifier
     * @param file
     * @throws IOException
     */
    public void printLabelings(Classifier classifier, File file) throws IOException {

        // Create a new iterator that will read raw instance data from
        //  the lines of a file.
        // Lines should be formatted as:
        //
        //   [name] [label] [data ... ]
        //
        //  in this case, "label" is ignored.

        CsvIterator reader =
                new CsvIterator(new FileReader(file),
                        "(\\w+)\\s+(\\w+)\\s+(.*)",
                        3, 2, 1);  // (data, label, name) field indices

        // Create an iterator that will pass each instance through
        //  the same pipe that was used to create the training data
        //  for the classifier.
        Iterator instances =
                classifier.getInstancePipe().newIteratorFrom(reader);

        // Classifier.classify() returns a Classification object
        //  that includes the instance, the classifier, and the
        //  classification results (the labeling). Here we only
        //  care about the Labeling.
        while (instances.hasNext()) {
            Labeling labeling = classifier.classify(instances.next()).getLabeling();

            // print the labels with their weights in descending order (ie best first)

            for (int rank = 0; rank < labeling.numLocations(); rank++){
                System.out.print(labeling.getLabelAtRank(rank) + ":" +
                        labeling.getValueAtRank(rank) + " ");
            }
            System.out.println();

        }
    }

    public void evaluate(Classifier classifier, File file) throws IOException {

        // Create an InstanceList that will contain the test data.
        // In order to ensure compatibility, process instances
        //  with the pipe used to process the original training
        //  instances.

        InstanceList testInstances = new InstanceList(classifier.getInstancePipe());

        // Create a new iterator that will read raw instance data from
        //  the lines of a file.
        // Lines should be formatted as:
        //
        //   [name] [label] [data ... ]

        CsvIterator reader =
                new CsvIterator(new FileReader(file),
                        "(\\w+)\\s+(\\w+)\\s+(.*)",
                        3, 2, 1);  // (data, label, name) field indices

        // Add all instances loaded by the iterator to
        //  our instance list, passing the raw input data
        //  through the classifier's original input pipe.

        testInstances.addThruPipe(reader);

        Trial trial = new Trial(classifier, testInstances);

        // The Trial class implements many standard evaluation
        //  metrics. See the JavaDoc API for more details.

        System.out.println("Accuracy: " + trial.getAccuracy());

        // precision, recall, and F1 are calcuated for a specific
        //  class, which can be identified by an object (usually
        //  a String) or the integer ID of the class

        System.out.println("F1 for class 'B': " + trial.getF1("B"));

        System.out.println("Precision for class '" +
                classifier.getLabelAlphabet().lookupLabel(1) + "': " +
                trial.getPrecision(1));
    }




}
