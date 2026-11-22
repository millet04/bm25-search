#include "CSRAlgorithms.h"

static void add_rows(CSRBase& model, const vector<vector<string>>& corpus, int kind, double k, double b, double delta) {
    for (size_t d=0; d<*model.doc_n; ++d) {
        const auto add_term = [&](const string& term_name, double f) {
            auto v=model.csr_vocabulary.find(term_name); if(v==model.csr_vocabulary.end()) return;
            double norm=1.0-b+b*model.dl[d]/model.avgdl, value=0.0;
            if(kind==11) value=f*(k+1.0)/(f+k*(model.dl[d]/model.avgdl));
            else if(kind==15) value=f*(k+1.0)/(f+k);
            else if(kind==25) value=f*(k+1.0)/(f+k*norm);
            else if(kind==26) value=f*(k+1.0)/(f+k*norm)+delta;
            else if(kind==27) { double c=f/norm; value=(k+1.0)*(c+delta)/(k+c+delta); }
            else value=f/model.dl[d];
            model.add_csr_value(d,v->second,value);
        };
        if (kind==26 || kind==27) {
            for (const auto& term : model.csr_vocabulary) {
                const auto found=model.freq[d].find(term.first);
                add_term(term.first, found==model.freq[d].end() ? 0.0 : static_cast<double>(found->second));
            }
        } else for (const auto& e : model.freq[d]) add_term(e.first, static_cast<double>(e.second));
        model.finish_csr_row();
    }
}
void BM11_CSR::set_model(const vector<vector<string>>& c,double k,double b,double delta){this->k=k;init_csr(c);add_rows(*this,c,11,k,0,0);}
void BM15_CSR::set_model(const vector<vector<string>>& c,double k,double b,double delta){this->k=k;init_csr(c);add_rows(*this,c,15,k,0,0);}
void BM25L_CSR::set_model(const vector<vector<string>>& c,double k,double b,double delta){this->k=k;this->b=b;this->delta=delta;init_csr(c);add_rows(*this,c,27,k,b,delta);}
void BM25Plus_CSR::set_model(const vector<vector<string>>& c,double k,double b,double delta){this->k=k;this->b=b;this->delta=delta;init_csr(c);add_rows(*this,c,26,k,b,delta);}
void TFIDF_CSR::set_model(const vector<vector<string>>& c){init_csr(c);add_rows(*this,c,0,0,0,0);}
