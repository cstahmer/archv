package org.eclipse.jetty.embedded;

import java.io.*;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;

public class ManyServlets 
{
  public static void main (String[] args) throws Exception
  {
    Server server = new Server(8080);
    ServletContextHandler context = new ServletContextHandler (ServletContextHandler.SESSIONS);
    context.addServlet (HelloServlet.class,"/*");
    context.addServlet (ProcessImages.class, "/process");
    context.addServlet (ScanDatabase.class, "/scan");
    context.addServlet (DrawMatches.class, "/draw");
    context.addServlet (ShowKeypoints.class, "/show");
    server.setHandler(context);

    server.start();
    server.join();
  }

  public static class HelloServlet extends HttpServlet
  {
    protected void doGet (HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException
    {
      response.setContentType ("text/html; charset=utf-8");
      response.setStatus (HttpServletResponse.SC_OK);
      response.getWriter().println("<h1>HOME PAGE</h1>");
      response.getWriter().println("<p>go to /process to process images</p>");
      response.getWriter().println("<p>go to /scan to scan the database</p>");
      response.getWriter().println("<p>go to /draw to print the matches</p>");
      response.getWriter().println("<p>go to /show to draw the keypoints</p>");
    }
  }

  public static class ProcessImages extends HttpServlet
  {
    protected void doGet (HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException
    {
      response.setContentType("text/html; charset=utf-8");
      response.setStatus(HttpServletResponse.SC_OK);
      response.getWriter().println("<h1> PROCESSING IMAGES </h1>");

      String filePath = "//home//arthur//opencv//jetty//demo//processImages.exe"; //need to //home// ...
      String [] arg = new String[6];
      arg[0] = "-i";
      arg[1] = "//home//arthur//imageset//small//"; //257 images here to be quick
      arg[2] = "-o";
      arg[3] = "//home//arthur//dummy//";
      arg[4] = "-p";
      arg[5] = "//home//arthur//opencv//parameters//param";

      try {
        ProcessBuilder builder = new ProcessBuilder (filePath, arg[0], arg[1], arg[2], arg[3], arg[4], arg[5]);
        Process process = builder.start();
        InputStream iStream = process.getInputStream();
        BufferedReader BR = new BufferedReader (new InputStreamReader(iStream), 1);

        String line;
        while ((line = BR.readLine()) != null) { //displays output
          System.out.println(line);
        }

      } catch (Exception ioe) {
          response.getWriter().println("<p>failed to read file</p>");
      }
      response.getWriter().println("<p> Process images into //home//arthur//dummy// </p>");
      response.getWriter().println("<p> {\"path\":\"//home//arthur//dummy//\"} </p>");
    }
  }

  public static class ScanDatabase extends HttpServlet
  {
    protected void doGet (HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException
    {
      response.setContentType("text/html; charset=utf-8");
      response.setStatus(HttpServletResponse.SC_OK);
      response.getWriter().println("<h1> SCANING DATABASE </h1>");

      String [] arg = new String[10];
      String filePath = "//home//arthur//programs//java//cpp//scan.exe"; //need to //home// ...
      arg[0] = "-i";
      arg[1] = "//home//arthur//programs//java//seed.jpg";
      arg[2] = "-d";
      arg[3] = "//home//arthur//imageset//small//";
      arg[4] = "-k";
      arg[5] = "//home//arthur//dummy//";
      arg[6] = "-o";
      arg[7] = "//home//arthur//programs//java//scan.jpg";
      arg[8] = "-p";
      arg[9] = "//home//arthur//opencv//parameters//param";

      try {
        ProcessBuilder builder = new ProcessBuilder (filePath, arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], arg[9]);
        Process process = builder.start();
        InputStream iStream = process.getInputStream();
        BufferedReader BR = new BufferedReader (new InputStreamReader(iStream), 1);

        String line;
        while ((line = BR.readLine()) != null) { //displays output
          System.out.println(line);
        }


      } catch (Exception ioe) {
          System.out.println("failed to read file");
        }
      response.getWriter().println("<p> Done with scan, matches saved in scan.txt </p>");
      response.getWriter().println("<p> {\"path\":\"//path//to//scan.jpg//\"} </p>");
    }
  }

  public static class DrawMatches extends HttpServlet
  {
    protected void doGet (HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException
    {
      response.setContentType("text/html; charset=utf-8");
      response.setStatus(HttpServletResponse.SC_OK);
      response.getWriter().println("<h1> DRAWING MATCHES </h1>");

      String [] arg = new String[10];
      String filePath = "//home//arthur//programs//java//cpp//draw.exe"; //need to //home// ...
      arg[0] = "-i1";
      arg[1] = "//home//arthur//programs//java//seed.jpg";
      arg[2] = "-12";
      arg[3] = "//home//arthur//programs//java//seed.jpg";
      arg[4] = "-o";
      arg[5] = "//home//arthur//programs//java//output.jpg";
      arg[6] = "-p";
      arg[7] = "//home//arthur//opencv//parameters//param";

    try {
      ProcessBuilder builder = new ProcessBuilder (filePath, arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7]);
      Process process = builder.start();
      InputStream iStream = process.getInputStream();
      BufferedReader BR = new BufferedReader (new InputStreamReader(iStream), 1);

      String line;
      while ((line = BR.readLine()) != null) { //displays output
        System.out.println(line);
      }


    } catch (Exception ioe) {
        System.out.println("failed to read file");
      }

    response.getWriter().println("<p> Done drawing Matches into output.jpg </p>");
    response.getWriter().println("<p> {\"path\":\"//path//to//output.jpg//\"} </p>");
    }
  }

  public static class ShowKeypoints extends HttpServlet
  {
    protected void doGet (HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException
    {
      response.setContentType("text/html; charset=utf-8");
      response.setStatus(HttpServletResponse.SC_OK);
      response.getWriter().println("<h1> SHOWING KEYPOINTS </h1>");

      String filePath = "//home//arthur//programs//java//cpp//show.exe"; //need to //home// ...
      String [] arg = new String[6];
      arg[0] = "-i";
      arg[1] = "//home//arthur//programs//java//seed.jpg";
      arg[2] = "-o";
      arg[3] = "//home//arthur//programs//java//keypoints.jpg";
      arg[4] = "-p";
      arg[5] = "//home//arthur//opencv//parameters//param";

    try {
      ProcessBuilder builder = new ProcessBuilder (filePath, arg[0], arg[1], arg[2], arg[3], arg[4], arg[5]);
      Process process = builder.start();
      InputStream iStream = process.getInputStream();
      BufferedReader BR = new BufferedReader (new InputStreamReader(iStream), 1);

      String line;
      while ((line = BR.readLine()) != null) { //displays output
        System.out.println(line);
      }

    } catch (Exception ioe) {
        System.out.println("failed to read file");
      }

    response.getWriter().println("<p> Done drawing Keypoints into keypoints.jpg </p>");
    response.getWriter().println("<p> {\"path\":\"//path//to//keypoints.jpg//\"} </p>");
    }
  }
}

     


