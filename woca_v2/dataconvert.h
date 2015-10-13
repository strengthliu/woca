#ifndef USE_H_
#define USE_H_

#include <iostream>
#include <windows.h>
//#include <afx.h>
#include <string>
#include <tchar.h>
using namespace std;
class CDataConverter
{
public:
	CDataConverter();
	virtual~ CDataConverter();
	char* WcharToChar(const wchar_t* wp);
	
	char* StringToChar(const string& s);
	char* WstringToChar(const wstring& ws);
	wchar_t* CharToWchar(const char* c);
	wchar_t* WstringToWchar(const wstring& ws);
	wchar_t* StringToWchar(const string& s);
	wstring StringToWstring(const string& s);
	string WstringToString(const wstring& ws);

	int WstringToInt(const wstring& ws);
	//string CStringToString(const CString& cs);
	//CString StringToCString(const string& s);
	void Release();
private:
	char* m_char;
	wchar_t* m_wchar;
};
#endif;