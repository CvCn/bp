import java.util.Arrays;

/**
 * https://blog.csdn.net/weixin_38347387/article/details/82936585
 */
public class BP {

    private double[] i = new double[2];

    private double[] h = new double[2];

    private double[] hw = new double[4];

    private double[] o = new double[2];

    private double[] ow = new double[4];

    private double[] b = new double[2];

    private double[] bw = new double[2];

    private double n = 0.5;

    enum node{
        h0,
        h1,
        o0,
        o1
    }

    public BP(){
        i[0] = 0.05;
        i[1] = 0.10;
        o[0] = 0.01;
        o[1] = 0.99;
        hw[0] = 0.15;
        hw[1] = 0.20;
        hw[2] = 0.25;
        hw[3] = 0.30;
        ow[0] = 0.40;
        ow[1] = 0.45;
        ow[2] = 0.50;
        ow[3] = 0.55;
        b[0] = 1;
        b[1] = 1;
        bw[0] = 0.35;
        bw[1] = 0.60;
    }

    public static void main(String[] args) {
        BP bp = new BP();
        bp.h[0] = bp.sigmoid(bp.predict(BP.node.h0));
        bp.h[1] = bp.sigmoid(bp.predict(BP.node.h1));

        double[] o = new double[]{
                bp.sigmoid(bp.predict(BP.node.o0)),
                bp.sigmoid(bp.predict(BP.node.o1))
        };

        double ow0_ = bp.fit(bp.ow[0], bp.f(bp.o[0], o[0], bp.h[0]));

        double ow1_ = bp.fit(bp.ow[1], bp.f(bp.o[0], o[0], bp.h[1]));

        double ow2_ = bp.fit(bp.ow[2], bp.f(bp.o[1], o[1], bp.h[0]));

        double ow3_ = bp.fit(bp.ow[3], bp.f(bp.o[1], o[1], bp.h[1]));

        System.out.println(ow2_);
    }

    public double fit(double source, double v){
        return source - n * v;
    }

    public double f(double target, double out, double proOut){
        return (out - target) * (out - Math.pow(out, 2)) * proOut;
    }


    public double sigmoid(double value){
        double e = 2.718281828459;
        return 1 / (1 + Math.pow(e, value * - 1));
    }

    public double predict(node node){
        double re = 0;
        switch (node){
            case h0:
                re = i[0] * hw[0] + i[1] * hw[1] + b[0] * bw[0];
                break;
            case h1:
                re = i[0] * hw[2] + i[1] * hw[3] + b[0] * bw[0];
                break;
            case o0:
                re = h[0] * ow[0] + h[1] * ow[1] + b[1] * bw[1];
                break;
            case o1:
                re = h[0] * ow[2] + h[1] * ow[3] + b[1] * bw[1];
                break;
        }

        return re;
    }


    public double ETotal(double[] out){
        double e0 = Math.pow(o[0] - out[0], 2)/2;
        double e1 = Math.pow(o[1] - out[1], 2)/2;
        return e0 + e1;
    }
}
