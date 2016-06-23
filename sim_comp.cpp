#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <sstream>
#include <set>
#include <stdlib.h>
#include <math.h>


#define LEN_BUF 50000000

using namespace std;

map <int,string> features_map;
typedef pair <int,string> ispair;
map <int,string> ::iterator isit_map;

float get_a(map<int,float> m1,map<int,float> m2)
{
	float a,value_f,value_f2;
	a=0.0;
	map <int,float>::iterator ifit_map,ifit_map2;
	ifit_map=m1.begin();
	int i;
	while (ifit_map!=m1.end())
	{
		i=ifit_map->first;
		ifit_map2=m2.find(i);
		if(ifit_map2!=m2.end())
		{
			value_f=ifit_map->second;
			value_f2=ifit_map2->second;
			a=a+value_f*value_f2;
		}
		ifit_map++;
	}

	return a;
}

double sim_computing(string v1,string v2)
{
	double sim;
	map <int,float> v1_map,v2_map;
	map <int,float>::iterator ifit_map,ifit_map2;
	typedef pair <int,float> ifpair;
	int head,tail,h;
	head=0;
	tail=v1.find(" ",head);
	string index_s,value_s,str;
	float value_f;
	int index_i;
	while(tail!=-1)
	{
		str=v1.substr(head,tail-head);
		h=str.find(":");
		index_s=str.substr(0,h);
		value_s=str.substr(h+1,str.length()-h-1);
		index_i=atoi(index_s.c_str());
		value_f=atof(value_s.c_str());
		v1_map.insert(ifpair(index_i,value_f));
		head=tail+1;
		tail=v1.find(" ",head);
	}

	head=0;
	tail=v2.find(" ",head);
	while(tail!=-1)
	{
		str=v2.substr(head,tail-head);
		h=str.find(":");
		index_s=str.substr(0,h);
		value_s=str.substr(h+1,str.length()-h-1);
		index_i=atoi(index_s.c_str());
		value_f=atof(value_s.c_str());
		v2_map.insert(ifpair(index_i,value_f));
		head=tail+1;
		tail=v2.find(" ",head);
	}
	float value_f2,a,b,c;
	b=0.0;
	c=0.0;

	if(v1_map.size()>v2_map.size())
		a=get_a(v2_map,v1_map);
	else
	    a=get_a(v1_map,v2_map);

	ifit_map=v1_map.begin();
	while(ifit_map!=v1_map.end())
	{
		value_f=ifit_map->second;
		value_f=value_f*value_f;
		b=b+value_f;
		ifit_map++;
	}
	b=sqrt(b);

	ifit_map=v2_map.begin();
	while(ifit_map!=v2_map.end())
	{
		value_f=ifit_map->second;
		value_f=value_f*value_f;
		c=c+value_f;
		ifit_map++;
	}
	c=sqrt(c);

	sim=a/(b*c);
	return sim;
}
int main(int argc, char **argv)
{
	if(argc!=3)
	{
		cerr<<"xx.exe feautre_file output_file"<<endl;
		return 1;
	}

	ifstream features_f(argv[1]);
	if(!features_f.is_open())
	{
		cerr<<"cannot open feautre_file"<<endl;
		return 1;
	}

	ofstream output_f(argv[2]);

	char * buf;
	buf=new char [LEN_BUF];
	string sentence;
	features_f.getline(buf,LEN_BUF);
	int i=0;
	while(!features_f.eof())
	{
		sentence=buf;
		features_map.insert(ispair(i,sentence));
		features_f.getline(buf,LEN_BUF);
		i++;
	}

	double ** matrix;
	int vector_count;

	vector_count=features_map.size();

	matrix=new double*[vector_count];
	for(i=0;i<vector_count;i++)
		matrix[i]=new double[vector_count];

	int j;
	string v1,v2;
	double sim;

	for(i=0;i<vector_count;i++)
	{
		cout<<"********"<<i<<endl;
		isit_map=features_map.find(i);
		v1=isit_map->second;
		for(j=i;j<vector_count;j++)
		{
			//cout<<j<<endl;
			isit_map=features_map.find(j);
			v2=isit_map->second;
			sim=sim_computing(v1,v2);
			matrix[i][j]=sim;
			if(i!=j)
				matrix[j][i]=sim;
		}
	}

	for(i=0;i<vector_count;i++)
	{
		for(j=0;j<vector_count;j++)
		{
			sim=matrix[i][j];
			if(j!=(vector_count-1))
				output_f<<sim<<"\t";
			else
				output_f<<sim<<endl;
		}
	}

	features_f.close();
	output_f.close();
	return 0;
}
