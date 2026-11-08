/*+ MergeJoin(k aka_t cn ct it1 mi t mc mk)
 MergeJoin(aka_t cn ct it1 mi t mc mk)
 MergeJoin(aka_t cn ct it1 mi t mc)
 HashJoin(cn ct it1 mi t mc)
 HashJoin(ct it1 mi t mc)
 HashJoin(it1 mi t mc)
 HashJoin(it1 mi t)
 HashJoin(mi t)
 IndexScan(k)
 IndexScan(aka_t)
 SeqScan(cn)
 SeqScan(ct)
 SeqScan(it1)
 IndexScan(mi)
 SeqScan(t)
 IndexScan(mc)
 IndexScan(mk)
 Leading((k ((aka_t (cn (ct ((it1 (mi t)) mc)))) mk))) */
SELECT MIN(mi.info) AS release_date,
       MIN(t.title) AS internet_movie
FROM aka_title AS aka_t,
     company_name AS cn,
     company_type AS ct,
     info_type AS it1,
     keyword AS k,
     movie_companies AS mc,
     movie_info AS mi,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code = '[us]'
  AND it1.info = 'release dates'
  AND mc.note LIKE '%(200%)%'
  AND mc.note LIKE '%(worldwide)%'
  AND mi.note LIKE '%internet%'
  AND mi.info LIKE 'USA:% 200%'
  AND t.production_year > 2000
  AND t.id = aka_t.movie_id
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = aka_t.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi.movie_id = aka_t.movie_id
  AND mc.movie_id = aka_t.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi.info_type_id
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id;
