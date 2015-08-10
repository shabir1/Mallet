package com.classifier;

import cc.mallet.classify.Classifier;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Labeling;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;

public class Main {

    public static void main(String args[]) throws IOException {

        Main mainClass = new Main();
        MyMalletClassifier myMalletClassifier = new MyMalletClassifier();
        InstanceList trainingInstanceList = new InstanceList(myMalletClassifier.getInstancePipe());

        trainingInstanceList.addThruPipe(new CsvIterator(new FileReader("resource/data.txt"),
                "(\\w+)\\s+(\\w+)\\s+(.*)",
                3, 2, 1));

        File file = new File("resource/data.txt");
        mainClass.printLabelings(myMalletClassifier.trainClassifier(trainingInstanceList),file);


    }
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


}
